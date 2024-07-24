#include <hailo/hailort.h>
#include <hailo/hailort_common.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>
#include <chrono>

#include "../output_tensor.h"
#include "../debug.h"
#include "allocator.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

// g++ -O2 -o yolov8-fps advanced/yolov8-fps.cpp -lhailort && ./yolov8-fps

std::string hefFile             = "yolov8m.hef";
std::string imgFilename         = "test-image-640x640.jpg";
float       confidenceThreshold = 0.5f;  // Lower number = accept more boxes
float       nmsIoUThreshold     = 0.45f; // Lower number = merge more boxes (I think!)
int         batchSize           = 8;

int run() {
	using namespace hailort;
	using namespace std::literals::chrono_literals;

	// Load image

	int            imgWidth = 0, imgHeight = 0, imgChan = 0;
	unsigned char* img_rgb_8 = stbi_load(imgFilename.c_str(), &imgWidth, &imgHeight, &imgChan, 3);
	if (!img_rgb_8) {
		printf("Failed to load image %s\n", imgFilename.c_str());
		return 1;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	// Load/Init
	////////////////////////////////////////////////////////////////////////////////////////////

	Expected<std::unique_ptr<VDevice>> vdevice_exp = VDevice::create();
	if (!vdevice_exp) {
		printf("Failed to create vdevice\n");
		return vdevice_exp.status();
	}
	std::unique_ptr<hailort::VDevice> vdevice = vdevice_exp.release();

	// Create infer model from HEF file.
	Expected<std::shared_ptr<InferModel>> infer_model_exp = vdevice->create_infer_model(hefFile);
	if (!infer_model_exp) {
		printf("Failed to create infer model\n");
		return infer_model_exp.status();
	}
	std::shared_ptr<hailort::InferModel> infer_model = infer_model_exp.release();
	infer_model->set_hw_latency_measurement_flags(HAILO_LATENCY_MEASURE);
	infer_model->set_batch_size(batchSize);
	infer_model->output()->set_nms_score_threshold(confidenceThreshold);
	infer_model->output()->set_nms_iou_threshold(nmsIoUThreshold);

	//printf("infer_model N inputs: %d\n", (int) infer_model->inputs().size());
	//printf("infer_model N outputs: %d\n", (int) infer_model->outputs().size());
	//printf("infer_model inputstream[0]: %s\n", DumpStream(infer_model->inputs()[0]).c_str());
	//printf("infer_model outputstream[0]: %s\n", DumpStream(infer_model->outputs()[0]).c_str());
	for (auto inputName : infer_model->get_input_names()) {
		printf("Input name: %s\n", inputName.c_str());
	}
	for (auto outputName : infer_model->get_output_names()) {
		printf("Output name: %s\n", outputName.c_str());
	}

	int nnWidth  = infer_model->inputs()[0].shape().width;
	int nnHeight = infer_model->inputs()[0].shape().height;

	//if (imgWidth * imgHeight * imgChan != input_frame_size) {
	//	printf("Input image resolution %d x %d x %d = %d not equal to NN input size %d", imgWidth, imgHeight, imgChan, int(imgWidth * imgHeight * imgChan), (int) input_frame_size);
	//}
	if (imgWidth != nnWidth || imgHeight != nnHeight) {
		printf("Input image resolution %d x %d not equal to NN input resolution %d x %d\n", imgWidth, imgHeight, nnWidth, nnHeight);
	}

	// Configure the infer model
	// infer_model->output()->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
	Expected<ConfiguredInferModel> configured_infer_model_exp = infer_model->configure();
	//auto configured_infer_model_exp = infer_model->configure();
	if (!configured_infer_model_exp) {
		printf("Failed to get configured infer model\n");
		return configured_infer_model_exp.status();
	}
	std::shared_ptr<hailort::ConfiguredInferModel> configured_infer_model = std::make_shared<ConfiguredInferModel>(configured_infer_model_exp.release());

	auto startTime = std::chrono::high_resolution_clock::now(); // this will be overwritten after the 1st run

	////////////////////////////////////////////////////////////////////////////////////////////
	// Run
	////////////////////////////////////////////////////////////////////////////////////////////

	const std::string& input_name       = infer_model->get_input_names()[0];
	size_t             input_frame_size = infer_model->input(input_name)->get_frame_size();
	//printf("input_name: %s\n", input_name.c_str());
	//printf("input_frame_size: %d\n", (int) input_frame_size); // eg 640x640x3 = 1228800

	size_t singleImageSize = imgWidth * imgHeight * imgChan;
	//uint8_t* img_batch       = (uint8_t*) malloc(singleImageSize * batchSize);
	//for (int i = 0; i < batchSize; i++) {
	//	memcpy(img_batch + i * singleImageSize, img_rgb_8, singleImageSize);
	//}

	int nRun = 10;

	for (int iRun = 0; iRun < nRun + 1; iRun++) {
		if (iRun == 1) {
			// Ignore the first run, which is much slower than the rest
			startTime = std::chrono::high_resolution_clock::now();
		}

		// Waiting for available requests in the pipeline.
		auto status = configured_infer_model->wait_for_async_ready(1s);
		if (status != HAILO_SUCCESS) {
			printf("Failed to wait for async ready, status = %d", (int) status);
			return status;
		}

		// Create infer bindings
		std::vector<ConfiguredInferModel::Bindings> bindings_batch;
		std::vector<OutTensor>                      output_tensors_batch;

		for (int i = 0; i < batchSize; i++) {
			Expected<ConfiguredInferModel::Bindings> bindings_exp = configured_infer_model->create_bindings();
			if (!bindings_exp) {
				printf("Failed to get infer model bindings\n");
				return bindings_exp.status();
			}

			status = bindings_exp->input(input_name)->set_buffer(MemoryView(img_rgb_8, input_frame_size));
			if (status != HAILO_SUCCESS) {
				printf("Failed to set memory buffer: %d\n", (int) status);
				return status;
			}

			// Output tensors.
			for (auto const& output_name : infer_model->get_output_names()) {
				size_t output_size = infer_model->output(output_name)->get_frame_size();
				//printf("output_size = %d\n", (int) output_size);

				uint8_t* output_buffer = (uint8_t*) malloc(output_size);
				if (!output_buffer) {
					printf("Could not allocate an output buffer!");
					return status;
				}

				status = bindings_exp->output(output_name)->set_buffer(MemoryView(output_buffer, output_size));
				if (status != HAILO_SUCCESS) {
					printf("Failed to set infer output buffer, status = %d", (int) status);
					return status;
				}

				const std::vector<hailo_quant_info_t> quant  = infer_model->output(output_name)->get_quant_infos();
				const hailo_3d_image_shape_t          shape  = infer_model->output(output_name)->shape();
				const hailo_format_t                  format = infer_model->output(output_name)->format();
				output_tensors_batch.emplace_back(output_buffer, output_name, quant[0], shape, format);
			}

			bindings_batch.emplace_back(std::move(bindings_exp.release()));
		}

		// Dispatch the job.
		Expected<AsyncInferJob> job_exp = configured_infer_model->run_async(bindings_batch, [](const AsyncInferCompletionInfo& completion_info) {
			// Use completion_info to get the async operation status
			// Note that this callback must be executed as quickly as possible
			(void) completion_info.status;
		});
		if (!job_exp) {
			printf("Failed to start async infer job, status = %d\n", (int) job_exp.status());
			return status;
		}
		hailort::AsyncInferJob job = job_exp.release();

		// Wait for job completion.
		status = job.wait(1s);
		if (status != HAILO_SUCCESS) {
			printf("Failed to wait for inference to finish, status = %d\n", (int) status);
			return status;
		}

		for (auto out : output_tensors_batch) {
			free(out.data);
		}
	}

	double elapsedSeconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
	int    nFrames        = nRun * batchSize;
	printf("%-16s %d\n", "Batch size", batchSize);
	printf("%-16s %s\n", "Model", hefFile.c_str());
	printf("%-16s %d x %d\n", "NN resolution", nnWidth, nnHeight);
	printf("%-16s %.2f\n", "FPS", nFrames / elapsedSeconds);
	printf("%-16s %.1fms\n", "Time per frame", 1000.0 * elapsedSeconds / nFrames);

	return 123456789;
}

int main(int argc, char** argv) {
	int status = run();
	if (status == 123456789)
		printf("SUCCESS\n");
	else
		printf("Failed with error code %d\n", status);
	return 0;
}
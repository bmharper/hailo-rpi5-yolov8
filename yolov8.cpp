#include <hailo/hailort.h>
#include <hailo/hailort_common.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>
#include <chrono>

#include "allocator.h"
#include "output_tensor.h"
#include "debug.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Use these invocations to build, or 'make'

// Release
// g++ -o yolohailo yolov8.cpp allocator.cpp -lhailort && ./yolohailo

// Debug
// g++ -g -O0 -o yolohailo yolov8.cpp allocator.cpp -lhailort && ./yolohailo

std::string hefFile             = "yolov8s.hef";
std::string imgFilename         = "test-image-640x640.jpg";
float       confidenceThreshold = 0.5f;

int run() {
	using namespace hailort;
	using namespace std::literals::chrono_literals;

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

	printf("infer_model N inputs: %d\n", (int) infer_model->inputs().size());
	printf("infer_model N outputs: %d\n", (int) infer_model->outputs().size());
	printf("infer_model inputstream[0]: %s\n", DumpStream(infer_model->inputs()[0]).c_str());
	printf("infer_model outputstream[0]: %s\n", DumpStream(infer_model->outputs()[0]).c_str());

	int nnWidth  = infer_model->inputs()[0].shape().width;
	int nnHeight = infer_model->inputs()[0].shape().height;

	// Configure the infer model
	// infer_model->output()->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
	Expected<ConfiguredInferModel> configured_infer_model_exp = infer_model->configure();
	if (!configured_infer_model_exp) {
		printf("Failed to get configured infer model\n");
		return configured_infer_model_exp.status();
	}
	std::shared_ptr<hailort::ConfiguredInferModel> configured_infer_model = std::make_shared<ConfiguredInferModel>(configured_infer_model_exp.release());

	// Create infer bindings
	Expected<ConfiguredInferModel::Bindings> bindings_exp = configured_infer_model->create_bindings();
	if (!bindings_exp) {
		printf("Failed to get infer model bindings\n");
		return bindings_exp.status();
	}
	hailort::ConfiguredInferModel::Bindings bindings = std::move(bindings_exp.release());

	////////////////////////////////////////////////////////////////////////////////////////////
	// Run
	////////////////////////////////////////////////////////////////////////////////////////////

	const std::string& input_name       = infer_model->get_input_names()[0];
	size_t             input_frame_size = infer_model->input(input_name)->get_frame_size();
	printf("input_name: %s\n", input_name.c_str());
	printf("input_frame_size: %d\n", (int) input_frame_size); // eg 640x640x3 = 1228800

	int            imgWidth = 0, imgHeight = 0, imgChan = 0;
	unsigned char* img_rgb_8 = stbi_load(imgFilename.c_str(), &imgWidth, &imgHeight, &imgChan, 3);
	if (!img_rgb_8) {
		printf("Failed to load image %s\n", imgFilename.c_str());
		return 1;
	}
	if (imgWidth * imgHeight * imgChan != input_frame_size) {
		printf("Input image resolution %d x %d x %d = %d not equal to NN input size %d", imgWidth, imgHeight, imgChan, int(imgWidth * imgHeight * imgChan), (int) input_frame_size);
	}
	if (imgWidth != nnWidth || imgHeight != nnHeight) {
		printf("Input image resolution %d x %d not equal to NN input resolution %d x %d\n", imgWidth, imgHeight, nnWidth, nnHeight);
	}

	auto status = bindings.input(input_name)->set_buffer(MemoryView((void*) (img_rgb_8), input_frame_size));
	if (status != HAILO_SUCCESS) {
		printf("Failed to set memory buffer: %d\n", (int) status);
		return status;
	}

	Allocator              allocator;
	std::vector<OutTensor> output_tensors;

	// Output tensors.
	for (auto const& output_name : infer_model->get_output_names()) {
		size_t output_size = infer_model->output(output_name)->get_frame_size();

		std::shared_ptr<uint8_t> output_buffer = allocator.Allocate(output_size);
		if (!output_buffer) {
			printf("Could not allocate an output buffer!");
			return status;
		}

		status = bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_size));
		if (status != HAILO_SUCCESS) {
			printf("Failed to set infer output buffer, status = %d", (int) status);
			return status;
		}

		const std::vector<hailo_quant_info_t> quant  = infer_model->output(output_name)->get_quant_infos();
		const hailo_3d_image_shape_t          shape  = infer_model->output(output_name)->shape();
		const hailo_format_t                  format = infer_model->output(output_name)->format();
		output_tensors.emplace_back(std::move(output_buffer), output_name, quant[0], shape, format);

		printf("Output tensor %s, %d bytes, shape (%d, %d, %d)\n", output_name.c_str(), (int) output_size, (int) shape.height, (int) shape.width, (int) shape.features);
		// printf("  %s\n", DumpFormat(format).c_str());
		for (auto q : quant) {
			printf("  Quantization scale: %f offset: %f\n", q.qp_scale, q.qp_zp);
		}
	}

	// Waiting for available requests in the pipeline.
	status = configured_infer_model->wait_for_async_ready(1s);
	if (status != HAILO_SUCCESS) {
		printf("Failed to wait for async ready, status = %d", (int) status);
		return status;
	}

	// Dispatch the job.
	Expected<AsyncInferJob> job_exp = configured_infer_model->run_async(bindings);
	if (!job_exp) {
		printf("Failed to start async infer job, status = %d\n", (int) job_exp.status());
		return status;
	}
	hailort::AsyncInferJob job = job_exp.release();

	// Detach and let the job run.
	job.detach();

	////////////////////////////////////////////////////////////////////////////////////////
	// Usually we'd go off and do something else at this point.
	////////////////////////////////////////////////////////////////////////////////////////

	// Prepare tensors for postprocessing.
	std::sort(output_tensors.begin(), output_tensors.end(), OutTensor::SortFunction);

	// Wait for job completion.
	status = job.wait(1s);
	if (status != HAILO_SUCCESS) {
		printf("Failed to wait for inference to finish, status = %d\n", (int) status);
		return status;
	}

	bool nmsOnHailo = infer_model->outputs().size() == 1 && infer_model->outputs()[0].is_nms();

	if (nmsOnHailo) {
		OutTensor* out = &output_tensors[0];

		const float* raw = (const float*) out->data.get();

		printf("Output shape: %d, %d\n", (int) out->shape.height, (int) out->shape.width);

		// The format is:
		// Number of boxes in that class (N), followed by the 5 box parameters, repeated N times
		size_t numClasses = (size_t) out->shape.height;
		size_t classIdx   = 0;
		size_t idx        = 0;
		while (classIdx < numClasses) {
			size_t numBoxes = (size_t) raw[idx++];
			for (size_t i = 0; i < numBoxes; i++) {
				float ymin       = raw[idx];
				float xmin       = raw[idx + 1];
				float ymax       = raw[idx + 2];
				float xmax       = raw[idx + 3];
				float confidence = raw[idx + 4];
				if (confidence >= 0.5f) {
					printf("class: %d, confidence: %.2f, %.0f,%.0f - %.0f,%.0f\n", classIdx, confidence, xmin * nnWidth, ymin * nnHeight, xmax * nnWidth, ymax * nnHeight);
				}
				idx += 5;
			}
			classIdx++;
		}
	} else {
		printf("No support in this example for NMS on CPU. See othe Hailo examples\n");
		return 1;
	}

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
#pragma once

#include <hailo/hailort.h>
#include <hailo/hailort_common.hpp>

std::string DumpShape(std::vector<size_t> shape) {
	std::string out;
	for (auto n : shape) {
		char el[100];
		sprintf(el, "%d,", (int) n);
		out += el;
	}
	return out;
}

std::string DumpShape(hailo_3d_image_shape_t shape) {
	char buf[1024];
	sprintf(buf, "(height: %d, width: %d, features: %d)", (int) shape.height, (int) shape.width, (int) shape.features);
	return buf;
}

std::string DumpFormat(hailo_format_t f) {
	char buf[1024];
	sprintf(buf, "(hailo_format = type: %s, order: %s, flags: %u)",
	        hailort::HailoRTCommon::get_format_type_str(f.type).c_str(),
	        hailort::HailoRTCommon::get_format_order_str(f.order).c_str(),
	        (unsigned) f.flags);
	return buf;
}

std::string DumpStream(const hailort::InferModel::InferStream& s) {
	char buf[1024];
	sprintf(buf, "'%s' %s %s, frame_size: %d bytes", s.name().c_str(), DumpShape(s.shape()).c_str(), DumpFormat(s.format()).c_str(), (int) s.get_frame_size());
	return buf;
}

// dump float32 as a 2d matrix
// stride is the number of float32 elements between rows.
// ncols is the number of columns that you want to print per line
// nrows is the number of rows that you want to print
std::string DumpFloat32(const float* out, int stride, int ncols, int nrows, float mul) {
	std::string result;
	for (int row = 0; row < nrows; row++) {
		int p = row * stride;
		for (int col = 0; col < ncols; col++) {
			char buf[1024];
			sprintf(buf, "%4.3f ", out[p + col] * mul);
			result += buf;
		}
		result += "\n";
	}
	return result;
}

// void DumpOutputTensor(
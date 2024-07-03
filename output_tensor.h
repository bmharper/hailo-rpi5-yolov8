#pragma once

#include <memory>
#include <string>

class OutTensor {
public:
	std::shared_ptr<uint8_t> data;
	std::string              name;
	hailo_quant_info_t       quant_info;
	hailo_3d_image_shape_t   shape;
	hailo_format_t           format;

	OutTensor(std::shared_ptr<uint8_t> data, const std::string& name, const hailo_quant_info_t& quant_info,
	          const hailo_3d_image_shape_t& shape, hailo_format_t format)
	    : data(std::move(data)), name(name), quant_info(quant_info), shape(shape), format(format) {
	}

	~OutTensor() {
	}

	//friend std::ostream& operator<<(std::ostream& os, const OutTensor& t) {
	//	os << "OutTensor: h " << t.shape.height << ", w " << t.shape.width << ", c " << t.shape.features;
	//	return os;
	//}

	static bool SortFunction(const OutTensor& l, const OutTensor& r) {
		return l.shape.width < r.shape.width;
	}
};

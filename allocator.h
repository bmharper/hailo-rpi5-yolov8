#pragma once

#include <memory>
#include <mutex>
#include <vector>
#include <stdlib.h>
#include <stdint.h>

class Allocator {
public:
	Allocator();
	~Allocator();

	void Reset();

	std::shared_ptr<uint8_t> Allocate(unsigned int size);

private:
	void free(uint8_t* ptr);

	struct AllocInfo {
		AllocInfo(uint8_t* _ptr, unsigned int _size, bool _free)
		    : ptr(_ptr), size(_size), free(_free) {
		}

		uint8_t*     ptr;
		unsigned int size;
		bool         free;
	};

	std::vector<AllocInfo> alloc_info_;
	std::mutex             lock_;
};

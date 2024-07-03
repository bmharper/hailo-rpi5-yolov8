#include "allocator.h"
#include <algorithm>
#include <string>
#include <sys/mman.h>

Allocator::Allocator() {
}

Allocator::~Allocator() {
	Reset();
}

void Allocator::Reset() {
	std::scoped_lock<std::mutex> l(lock_);

	for (auto& info : alloc_info_)
		munmap(info.ptr, info.size);

	alloc_info_.clear();
}

std::shared_ptr<uint8_t> Allocator::Allocate(unsigned int size) {
	std::scoped_lock<std::mutex> l(lock_);
	uint8_t*                     ptr = nullptr;

	auto info = std::find_if(alloc_info_.begin(), alloc_info_.end(),
	                         [size](const AllocInfo& info) { return info.free && info.size == size; });
	if (info != alloc_info_.end()) {
		info->free = false;
		ptr        = info->ptr;
	}

	if (!ptr) {
		void* addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
		if (addr == MAP_FAILED)
			return {};

		ptr = static_cast<uint8_t*>(addr);
		alloc_info_.emplace_back(ptr, size, false);
	}

	return std::shared_ptr<uint8_t>(ptr, [this](uint8_t* ptr) { this->free(ptr); });
}

void Allocator::free(uint8_t* ptr) {
	std::scoped_lock<std::mutex> l(lock_);

	auto info = std::find_if(alloc_info_.begin(), alloc_info_.end(),
	                         [ptr](const AllocInfo& info) { return info.ptr == ptr; });
	if (info != alloc_info_.end())
		info->free = true;
}

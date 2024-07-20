#pragma once

#include <vector>
#include <stdlib.h>
#include <sys/mman.h>

// A very simple page-aligned memory heap.
// DO NOT use this if you are allocating more than a handful of buffers.
// All operations are O(n) where n is the number of buffers allocated.
// We only reuse buffers when the requested size matches 100%.
// So this is intended for frequent re-use where the buffer sizes are consistent.
class PageAlignedAllocator {
public:
	struct Buffer {
		void*  P;
		size_t Size;
	};
	std::vector<Buffer> Used;
	std::vector<Buffer> Available;

	~PageAlignedAllocator() {
		for (auto b : Used)
			munmap(b.P, b.Size);
		for (auto b : Available)
			munmap(b.P, b.Size);
	}

	void* Alloc(size_t size) {
		for (size_t i = 0; i < Available.size(); i++) {
			if (Available[i].Size == size) {
				void* p = Available[i].P;
				Used.push_back(Available[i]);
				std::swap(Available[i], Available.back());
				Available.pop_back();
				return p;
			}
		}
		Buffer b;
		b.P    = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		b.Size = size;
		Used.push_back(b);
		return b.P;
	}

	void Free(void* buf) {
		for (size_t i = 0; i < Used.size(); i++) {
			if (Used[i].P == buf) {
				Available.push_back(Used[i]);
				std::swap(Used[i], Used.back());
				Used.pop_back();
				return;
			}
		}
	}
};
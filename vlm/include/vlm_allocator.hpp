#pragma once

namespace vlm {

struct Allocator {
    Allocator() = default;
    virtual ~Allocator() = default;

    virtual void* malloc(unsigned long long size) const = 0;
    virtual void free(void* ptr) const = 0;
};

} // namespace vlm
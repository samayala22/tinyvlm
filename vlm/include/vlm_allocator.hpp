#pragma once

namespace vlm {

typedef void* (*malloc_f)(size_t size);
typedef void (*free_f)(void* ptr);
typedef void* (*memcpy_f)(void* dst, const void* src, size_t size);
typedef void* (*memset_f)(void* dst, int value, size_t size);

struct Allocator {
    malloc_f h_malloc;
    malloc_f d_malloc;
    free_f h_free;
    free_f d_free;
    memcpy_f hh_memcpy;
    memcpy_f hd_memcpy;
    memcpy_f dh_memcpy;
    memcpy_f dd_memcpy;
    memset_f h_memset;
    memset_f d_memset;
};

} // namespace vlm
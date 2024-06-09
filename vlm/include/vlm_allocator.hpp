#pragma once

namespace vlm {

typedef void* (*malloc_f)(unsigned long long size);
typedef void (*free_f)(void* ptr);
typedef void* (*memcpy_f)(void* dst, const void* src, unsigned long long size);
typedef void* (*memset_f)(void* dst, int value, unsigned long long size);

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
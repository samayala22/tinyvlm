#pragma once

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace vlm {

using malloc_f = void* (*)(std::size_t size);
using free_f = void (*)(void* ptr);
using memcpy_f = void* (*)(void* dst, const void* src, std::size_t size);
using memset_f = void* (*)(void* dst, int value, std::size_t size);

struct Allocator {
    const malloc_f h_malloc;
    const malloc_f d_malloc;
    const free_f h_free;
    const free_f d_free;
    const memcpy_f hh_memcpy;
    const memcpy_f hd_memcpy;
    const memcpy_f dh_memcpy;
    const memcpy_f dd_memcpy;
    const memset_f h_memset;
    const memset_f d_memset;

    Allocator() = delete;
    Allocator(
        const malloc_f h_malloc,
        const malloc_f d_malloc,
        const free_f h_free,
        const free_f d_free,
        const memcpy_f hh_memcpy,
        const memcpy_f hd_memcpy,
        const memcpy_f dh_memcpy,
        const memcpy_f dd_memcpy,
        const memset_f h_memset,
        const memset_f d_memset
    ) : h_malloc(h_malloc), d_malloc(d_malloc), h_free(h_free), d_free(d_free),
        hh_memcpy(hh_memcpy), hd_memcpy(hd_memcpy), dh_memcpy(dh_memcpy), dd_memcpy(dd_memcpy),
        h_memset(h_memset), d_memset(d_memset) {}
};

struct Host {};
struct Device {};
struct HostDevice {};

// template<typename T, typename Location>
// class Buffer {
//     static_assert(std::is_same_v<Location, Host> || std::is_same_v<Location, Device> || std::is_same_v<Location, HostDevice>, "Invalid Location type");
//     using is_host = std::bool_constant<std::is_same_v<Location, Host> || std::is_same_v<Location, HostDevice>>;
//     using is_device = std::bool_constant<std::is_same_v<Location, Device> || std::is_same_v<Location, HostDevice>>;

// public:
//     explicit Buffer(Allocator* allocator, std::size_t n) : allocator(allocator) { alloc(n); }
//     explicit Buffer() = default;
//     ~Buffer() { dealloc(); }

//     T* h_ptr() {
//         static_assert(is_host::value);
//         return _host;
//     }
//     const T* h_ptr() const {
//         static_assert(is_host::value);
//         return _host;
//     }
//     T* d_ptr() {
//         static_assert(is_device::value);
//         return _device;
//     }
//     const T* d_ptr() const {
//         static_assert(is_device::value);
//         return _device;
//     }

//     void to_device() {
//         static_assert(is_host::value && is_device::value);
//         if (allocator->d_malloc != allocator->h_malloc) 
//             allocator->hd_memcpy(_device, static_cast<void*>(_host), _size * sizeof(T));
//     }

//     void to_host() {
//         static_assert(is_host::value && is_device::value);
//         if (allocator->d_malloc != allocator->h_malloc) 
//             allocator->dh_memcpy(_host, static_cast<void*>(_device), _size * sizeof(T));
//     }

//     void set_allocator(Allocator* allocator) {
//         this->allocator = allocator;
//     }

// private:
//     // 32 bytes max
//     Allocator* allocator = nullptr;
//     T* _host = nullptr;
//     T* _device = nullptr;
//     std::size_t _size = 0;

//     void alloc(std::size_t n) {
//         assert(allocator);
//         _size = n;
//         if constexpr (is_host::value) {
//             _host = static_cast<T*>(allocator->h_malloc(_size * sizeof(T)));
//         }
//         if constexpr (is_device::value) {
//             if constexpr (!is_host::value) assert(allocator->d_malloc != allocator->h_malloc);
//             _device = (allocator->d_malloc != allocator->h_malloc) 
//                 ? static_cast<T*>(allocator->d_malloc(_size * sizeof(T))) 
//                 : _host;
//         }
//     }

//     void dealloc() {
//         if (allocator) {
//             if constexpr (is_host::value) {
//                 allocator->h_free(_host);
//             }
//             if constexpr (is_device::value) {
//                 if (allocator->d_malloc != allocator->h_malloc) allocator->d_free(_device);
//             }
//         }
//     }

//     Buffer(Buffer&&) = delete;
//     Buffer(const Buffer&) = delete;
//     Buffer& operator=(Buffer&&) = delete;
//     Buffer& operator=(const Buffer&) = delete;
// };

template<typename T, typename Location>
class Buffer {
    static_assert(std::is_same_v<Location, Host> || std::is_same_v<Location, Device> || std::is_same_v<Location, HostDevice>, "Invalid Location type");
    using is_host = std::bool_constant<std::is_same_v<Location, Host> || std::is_same_v<Location, HostDevice>>;
    using is_device = std::bool_constant<std::is_same_v<Location, Device> || std::is_same_v<Location, HostDevice>>;

public:
    Buffer() = delete;
    explicit Buffer(const Allocator& allocator, std::size_t n) : allocator(allocator) { alloc(n); }
    explicit Buffer(const Allocator& allocator) : allocator(allocator) {}
    ~Buffer() { dealloc(); }

    T* h_ptr() {
        static_assert(is_host::value);
        return _host;
    }
    const T* h_ptr() const {
        static_assert(is_host::value);
        return _host;
    }
    T* d_ptr() {
        static_assert(is_device::value);
        return _device;
    }
    const T* d_ptr() const {
        static_assert(is_device::value);
        return _device;
    }

    void to_device() {
        static_assert(is_host::value && is_device::value);
        if (allocator.d_malloc != allocator.h_malloc) 
            allocator.hd_memcpy(_device, static_cast<void*>(_host), _size * sizeof(T));
    }

    void to_host() {
        static_assert(is_host::value && is_device::value);
        if (allocator.d_malloc != allocator.h_malloc) 
            allocator.dh_memcpy(_host, static_cast<void*>(_device), _size * sizeof(T));
    }

    void set_allocator(Allocator* allocator) {
        this->allocator = allocator;
    }

    std::size_t len() const { return _size; }
    std::size_t size() const { return _size * sizeof(T); }

    void alloc(std::size_t n) {
        _size = n;
        if constexpr (is_host::value) {
            _host = static_cast<T*>(allocator.h_malloc(_size * sizeof(T)));
        }
        if constexpr (is_device::value) {
            if constexpr (!is_host::value) assert(allocator.d_malloc != allocator.h_malloc);
            _device = (allocator.d_malloc != allocator.h_malloc) 
                ? static_cast<T*>(allocator.d_malloc(_size * sizeof(T))) 
                : _host;
        }
    }

    void dealloc() {
        if constexpr (is_host::value) {
            allocator.h_free(_host);
        }
        if constexpr (is_device::value) {
            if (allocator.d_malloc != allocator.h_malloc) allocator.d_free(_device);
        }
    }

private:
    // 32 bytes max
    const Allocator& allocator;
    T* _host = nullptr;
    T* _device = nullptr;
    std::size_t _size = 0;

    Buffer(Buffer&&) = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

} // namespace vlm
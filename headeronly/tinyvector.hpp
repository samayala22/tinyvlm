#pragma once

#include <cassert>
#include <cstdlib>
#include <vector>

namespace tiny {

inline void* aligned_malloc(size_t size, size_t alignment) {
    assert(((alignment & (alignment - 1)) == 0) && "alignment must be a power of two");
    assert((alignment >= sizeof(void*)) && "alignment must be at least the size of a pointer");
    void* res = nullptr;
#ifdef _WIN32
    res = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&res, alignment, size) != 0) res = nullptr;
#endif
    return res;
}

inline void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

template <typename T, size_t Alignment>
class AlignedAllocator {
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // Rebind mechanism
    template<typename U>
    struct rebind {
        typedef AlignedAllocator<U, Alignment> other;
    };

    AlignedAllocator() = default;

    template <typename U, size_t N>
    AlignedAllocator(const AlignedAllocator<U, N>&) {}

    T* allocate(std::size_t n) {
        return static_cast<T*>(aligned_malloc(n * sizeof(T), Alignment));
    }

    void deallocate(T* p, std::size_t) {
        aligned_free(p);
    }

    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new(p) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) {
        p->~U();
    }
};

template<typename T, size_t Alignment = 64>
using vector = std::vector<T, AlignedAllocator<T, Alignment>>;

}
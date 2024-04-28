#pragma once

#include <utility>
#include <cstdint>
#include <vcruntime.h>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <memory>

#include "linalg.h"

namespace vlm {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
 
using f32 = float;
using f64 = double;

static const f64 PI_d = 3.14159265358979;
static const f32 PI_f = 3.141593f;
static const f64 EPS_d = 2.22045e-16;
static const f32 EPS_f = 1.19209e-07f;
static const f64 EPS_sqrt_d = std::sqrt(EPS_d);
static const f32 EPS_sqrt_f = std::sqrt(EPS_f);
static const f64 EPS_cbrt_d = std::cbrt(EPS_d);
static const f32 EPS_cbrt_f = std::cbrt(EPS_f);

// Optimized n power function that uses template folding optimisations
// to generate a series of mul instructions
// https://godbolt.org/z/4boYoM3e8
template <int N, typename T>
inline T pow(T x) {
    T result = x;
    for (int i = 1; i < N; ++i) result *= x;
    return result;
}

// Structure of arrays struct to store 3D data (coordinates or vectors)
template<typename T>
struct SoA_3D_t {
    std::vector<T> x = {};
    std::vector<T> y = {};
    std::vector<T> z = {};
    u64 size = 0;
    
    void resize(u64 size_) {
        size = size_;
        x.resize(size);
        y.resize(size);
        z.resize(size);
    }

    void reserve(u64 size_) {
        size = size_;
        x.reserve(size);
        y.reserve(size);
        z.reserve(size);
    }
};

template<typename T, size_t N>
class StackArray {
    T data[N];
};

template<typename T>
class HeapArray {
    T* first;
    T* last;
    T* end;
};

template<typename T, size_t N>
class View {
    private:
        const T* data;
        const StackArray<T, N> dims;
        const StackArray<T, N> strides;
    public:
        T* operator()() {
            return data;
        }
        template<typename... Idx>
        const T& operator()(Idx... idx) const {
            static_assert(sizeof...(idx) == N, "The number of indices must match the dimension N.");
            const size_t indices[N] = {idx...};
            size_t index = 0;
            for (size_t i = 0; i < N; ++i) {
                index += indices[i] * strides.data[i];
            }
            return data[index];
        }
};

} // namespace vlm
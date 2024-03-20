#pragma once

#include <utility>
#include <cstdint>
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

template<typename T>
struct SoA3D_t {
    T* x = nullptr;
    T* y = nullptr;
    T* z = nullptr;
    u64 size = 0;
};

} // namespace vlm
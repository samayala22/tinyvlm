#pragma once

#include <utility>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <limits>

#include "linalg.h"
#include "tinyconfig.hpp"

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

constexpr f64 PI_d = 3.14159265358979;
constexpr f32 PI_f = 3.141593f;
constexpr f64 EPS_d = std::numeric_limits<f64>::epsilon();
constexpr f32 EPS_f = std::numeric_limits<f32>::epsilon();

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

} // namespace vlm
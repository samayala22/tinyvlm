#pragma once

#include "vlm_types.hpp"
#include "linalg.h"
#include <tuple>

// List of pure functions 
namespace vlm {

template<typename... Iters>
class zip_iterator {
    std::tuple<Iters...> iters;
    
public:
    using value_type = std::tuple<typename std::iterator_traits<Iters>::reference...>;
    
    explicit zip_iterator(Iters... its) : iters(its...) {}
    
    value_type operator*() {
        return std::apply([](auto&... its) { 
            return value_type(*its...); 
        }, iters);
    }
    
    zip_iterator& operator++() {
        std::apply([](auto&... its) { ((++its), ...); }, iters);
        return *this;
    }
    
    bool operator!=(const zip_iterator& other) const {
        return std::get<0>(iters) != std::get<0>(other.iters);
    }
};

template<typename... Containers>
class zip_helper {
    std::tuple<Containers&...> containers;
    
public:
    explicit zip_helper(Containers&... cs) : containers(cs...) {}
    
    auto begin() {
        return std::apply([](auto&... conts) {
            return zip_iterator(std::begin(conts)...);
        }, containers);
    }
    
    auto end() {
        return std::apply([](auto&... conts) {
            return zip_iterator(std::end(conts)...);
        }, containers);
    }
};

template<typename... Containers>
auto zip(Containers&... containers) {
    return zip_helper<Containers...>(containers...);
}

linalg::float3 compute_freestream(const f32 u_inf, const f32 alpha, const f32 beta);
linalg::float3 compute_lift_axis(const linalg::float3& freestream_);
linalg::float3 compute_stream_axis(const linalg::float3& freestream_);

f32 to_degrees(f32 radians);
f32 to_radians(f32 degrees);

template<typename T>
constexpr T pow(T base, int exponent) {
    T result = 1;
    while (exponent > 0) {
        if (exponent & 1) {
            result *= base;
        }
        base *= base;
        exponent >>= 1;
    }
    return result;
}

} // namespace vlm
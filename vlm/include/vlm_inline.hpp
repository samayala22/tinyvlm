#pragma once

#include "vlm_types.hpp"

// List of pure functions 
namespace vlm {

inline Eigen::Vector3f compute_freestream(f32 u_inf, f32 alpha, f32 beta) {
    return Eigen::Vector3f{
        u_inf * std::cos(alpha) * std::cos(beta),
        - u_inf * std::cos(alpha) * std::sin(beta),
        u_inf * std::sin(alpha),
    };
}

inline Eigen::Vector3f compute_lift_axis(const Eigen::Vector3f& freestream_) {
    return (freestream_.cross(Eigen::Vector3f::UnitY()).normalized());
}

inline Eigen::Vector3f compute_stream_axis(const Eigen::Vector3f& freestream_) {
    return (freestream_.normalized());
}

}
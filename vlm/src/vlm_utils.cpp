#include "vlm_utils.hpp"

using namespace vlm;

linalg::alias::float3 vlm::compute_freestream(const f32 u_inf, const f32 alpha, const f32 beta) {
    return linalg::alias::float3{
        u_inf * std::cos(alpha) * std::cos(beta),
        - u_inf * std::cos(alpha) * std::sin(beta),
        u_inf * std::sin(alpha),
    };
}

linalg::alias::float3 vlm::compute_lift_axis(const linalg::alias::float3& freestream_) {
    return linalg::normalize(linalg::cross(freestream_, {0.f, 1.f, 0.f}));
}

linalg::alias::float3 vlm::compute_stream_axis(const linalg::alias::float3& freestream_) {
    return linalg::normalize(freestream_);
}

f32 vlm::to_degrees(f32 radians) {
    return radians * 180.0f / PI_f;
}

f32 vlm::to_radians(f32 degrees) {
    return degrees * PI_f / 180.0f;
}
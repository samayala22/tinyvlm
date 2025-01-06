#include "vlm_utils.hpp"

using namespace vlm;

linalg::float3 vlm::compute_freestream(const f32 u_inf, const f32 alpha, const f32 beta) {
    return linalg::float3{
        u_inf * std::cos(alpha) * std::cos(beta),
        - u_inf * std::cos(alpha) * std::sin(beta),
        u_inf * std::sin(alpha),
    };
}

linalg::float3 vlm::compute_lift_axis(const linalg::float3& freestream_) {
    return linalg::normalize(linalg::cross(freestream_, {0.f, 1.f, 0.f}));
}

linalg::float3 vlm::compute_stream_axis(const linalg::float3& freestream_) {
    return linalg::normalize(freestream_);
}
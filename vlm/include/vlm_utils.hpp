#pragma once

#include "vlm_types.hpp"

// List of pure functions 
namespace vlm {

linalg::alias::float3 compute_freestream(const f32 u_inf, const f32 alpha, const f32 beta);
linalg::alias::float3 compute_lift_axis(const linalg::alias::float3& freestream_);
linalg::alias::float3 compute_stream_axis(const linalg::alias::float3& freestream_);

}
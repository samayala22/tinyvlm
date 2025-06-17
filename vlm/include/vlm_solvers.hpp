#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"
#include "vlm_memory.hpp"

namespace vlm {

void anderson_acceleration(
    Backend* backend,
    const TensorView1dD& x0,
    const std::function<void(const TensorView1dD& x, const TensorView1dD& y)>& f,
    i32 max_iter = 100,
    f64 tol_res = 1e-6,
    i32 m = 3
);

void anderson_acceleration(
    Backend* backend,
    const TensorView1fD& x0,
    const std::function<void(const TensorView1fD& x, const TensorView1fD& y)>& f,
    i32 max_iter = 100,
    f32 tol_res = 1e-6,
    i32 m = 3
);
    
} // namespace vlm
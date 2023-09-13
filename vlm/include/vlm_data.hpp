#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"

namespace vlm {

struct Data {
    std::vector<f32> gamma = {}; // unknowns to solve for (vortices)
    std::vector<f32> delta_gamma = {}; // gamma_i_j - gamma_i-1_j
    f32 cl = 0.0f;
    f32 cd = 0.0f;
    f32 cm = 0.0f;
    
    Vec3 u_inf; // freestream velocity
    Vec3 lift_axis; // lift axis
    Vec3 ref_pt = {0.25f, 0.0f, 0.0f}; // reference point (for moment computation

    f32 rho = 1.0f;
    f32 s_ref = 3.25f;

    u32 nc = 0;
    u32 ns = 0;
    void alloc(const u32 nc_, const u32 ns_);
    void reset();
    void compute_freestream(f32 alpha);
    void postprocess();
};

} // namespace vlm
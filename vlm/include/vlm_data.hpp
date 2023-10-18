#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"

namespace vlm {

struct Data {
    std::vector<f32> gamma = {}; // unknowns to solve for (vortices)
    std::vector<f32> delta_gamma = {}; // gamma_i_j - gamma_i-1_j
    f32 cl = 0.0f;
    f32 cd = 0.0f;
    f32 cm_x = 0.0f;
    f32 cm_y = 0.0f;
    f32 cm_z = 0.0f;
    
    Vec3<f32> u_inf; // freestream velocity (computed in compute_freestream)
    Vec3<f32> lift_axis; // lift axis (computed in compute_freestream)
    Vec3<f32> ref_pt = {0.25f, 0.0f, 0.0f}; // reference point (for moment computation

    f32 rho = 1.0f;
    f32 s_ref = 3.25f; // reference area (of wing)
    f32 c_ref = 0.85f; // reference chord (of wing) ~= 0.5 * (c_root + c_tip)
    f32 sigma_vatistas = 0.0f; // vatistas coefficient

    void alloc(const u32 size);
    void reset();
    void compute_freestream(f32 alpha);
    Data() = default;
    Data(tiny::Config& cfg);
};

} // namespace vlm
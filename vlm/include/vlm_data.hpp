#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"

namespace vlm {

struct Data {
    std::vector<f32> gamma = {}; // unknowns to solve for (vortex strengths))
    std::vector<f32> delta_gamma = {};
    f32 cl = 0.0f; // lift coefficient
    f32 cd = 0.0f; // drag coefficient
    Eigen::Vector3f cm = {}; // moment coefficients
    
    f32 alpha = 0.0f; // global angle of attack
    f32 beta = 0.0f; // global sideslip angle
    f32 u_inf = 1.0f; // freestream velocity magnitude
    Eigen::Vector3f ref_pt = {0.25f, 0.0f, 0.0f}; // reference point (for moment computation

    f32 rho = 1.0f;
    f32 s_ref = 0.0f; // reference area (of wing)
    f32 c_ref = 0.0f; // reference chord (of wing)
    f32 sigma_vatistas = 0.0f; // vatistas coefficient

    void alloc(const u32 size);
    void reset();
    
    // Global values
    Eigen::Vector3f freestream() const;
    Eigen::Vector3f lift_axis() const;
    Eigen::Vector3f stream_axis() const;
    // Local values
    Eigen::Vector3f freestream(f32 alpha, f32 beta) const;
    Eigen::Vector3f lift_axis(const Eigen::Vector3f& freestream_) const;
    Eigen::Vector3f stream_axis(const Eigen::Vector3f& freestream_) const;

    Data() = default;
    Data(tiny::Config& cfg);
};

// Light Class for local flow characteristics
class FlowData {
    public:
    const f32 alpha;
    const f32 beta;
    const f32 u_inf;
    const f32 rho;
    const Eigen::Vector3f freestream;
    const Eigen::Vector3f lift_axis;
    const Eigen::Vector3f stream_axis;

    FlowData(const f32 alpha_, const f32 beta_, const f32 u_inf_, const f32 rho_);
    private:
    Eigen::Vector3f f_freestream(f32 alpha, f32 beta) const;
    Eigen::Vector3f f_lift_axis(const Eigen::Vector3f& freestream_) const;
    Eigen::Vector3f f_stream_axis(const Eigen::Vector3f& freestream_) const;
};

} // namespace vlm
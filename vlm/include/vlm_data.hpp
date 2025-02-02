#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"
#include "vlm_memory.hpp"
#include "tinyinterpolate.hpp"
#include "linalg.h"

namespace vlm {

// Flow characteristics
class FlowData {
    public:
    // TODO: maybe move these to methods
    const f32 alpha; // global angle of attack
    const f32 beta; // global sideslip angle
    const f32 u_inf; // freestream velocity magnitude
    const f32 rho; // fluid density

    const linalg::float3 freestream;
    const linalg::float3 lift_axis;
    const linalg::float3 stream_axis;

    FlowData(const f32 alpha_, const f32 beta_, const f32 u_inf_, const f32 rho_);
    FlowData(const linalg::float3& freestream_, const f32 rho_);
};

class AeroCoefficients {
    public:
    const f32 cl;
    const f32 cd;
    const linalg::float3 cm;

    AeroCoefficients(const f32 cl_, const f32 cd_, const linalg::float3& cm_) :
        cl(cl_),
        cd(cd_),
        cm(cm_)
    {}
};

template<class Interpolator>
class WingProfile {
    static_assert(std::is_base_of<tiny::Interpolator<f32>, Interpolator>::value, "Invalid interpolator type");

    public:
    const std::vector<f32> alphas;
    const std::vector<f32> CL;
    const std::vector<f32> CD;
    const std::vector<f32> CMx;
    const std::vector<f32> CMy;
    const std::vector<f32> CMz;

    const Interpolator CL_interpolator;
    const Interpolator CD_interpolator;
    const Interpolator CMx_interpolator;
    const Interpolator CMy_interpolator;
    const Interpolator CMz_interpolator;

    WingProfile(
        const std::vector<f32>& alphas,
        const std::vector<f32>& CL,
        const std::vector<f32>& CD,
        const std::vector<f32>& CMx,
        const std::vector<f32>& CMy,
        const std::vector<f32>& CMz
    ) :
        CL(CL),
        CD(CD),
        CMx(CMx),
        CMy(CMy),
        CMz(CMz),
        CL_interpolator(alphas, CL),
        CD_interpolator(alphas, CD),
        CMx_interpolator(alphas, CMx),
        CMy_interpolator(alphas, CMy),
        CMz_interpolator(alphas, CMz)
    {}
};

template<typename Interpolator>
class Database2D {
    public:
    std::vector<WingProfile<Interpolator>> profiles; // profiles
    std::vector<f32> profiles_pos; // y position of profiles
    Database2D() = default;
    f32 interpolate_CL(f32 alpha, f32 y) const {
        // TODO: perform linear blending between the two profiles closest to y
        const auto& profile = profiles[0];
        const f32 CL = profile.CL_interpolator(alpha);
        return CL;
    }
};

using Database = Database2D<tiny::AkimaInterpolator<f32>>;

} // namespace vlm
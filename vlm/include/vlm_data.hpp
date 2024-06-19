#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"
#include "vlm_allocator.hpp"
#include "tinyinterpolate.hpp"

namespace vlm {

struct Data {
    f32* lhs = nullptr; // (ns*nc)^2
    f32* rhs = nullptr; // ns*nc
    f32* gamma = nullptr; // (nc+nw)*ns
    f32* gamma_prev = nullptr; // nc*ns
    f32* delta_gamma = nullptr; // nc*ns
    f32* rollup_vertices = nullptr; // (nc+nw+1)*(ns+1)*3
    f32* local_velocities = nullptr; // ns*nc*3
    f32* trefftz_buffer = nullptr; // ns TODO: can we get rid of this ??
};

void data_alloc(const malloc_f malloc, Data* data, u64 nc, u64 ns, u64 nw);
void data_free(const free_f free, Data* data);

// Flow characteristics
class FlowData {
    public:
    // TODO: maybe move these to methods
    const f32 alpha; // global angle of attack
    const f32 beta; // global sideslip angle
    const f32 u_inf; // freestream velocity magnitude
    const f32 rho; // fluid density

    const linalg::alias::float3 freestream;
    const linalg::alias::float3 lift_axis;
    const linalg::alias::float3 stream_axis;

    FlowData(const f32 alpha_, const f32 beta_, const f32 u_inf_, const f32 rho_);
    FlowData(const linalg::alias::float3& freestream_, const f32 rho_);
};

class AeroCoefficients {
    public:
    const f32 cl;
    const f32 cd;
    const linalg::alias::float3 cm;

    AeroCoefficients(const f32 cl_, const f32 cd_, const linalg::alias::float3& cm_) :
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
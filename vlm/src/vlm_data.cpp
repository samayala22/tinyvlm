#include "vlm_data.hpp"
#include "vlm_types.hpp"

using namespace vlm;

void Data::alloc(const u32 nc_, const u32 ns_) {
        nc = nc_;
        ns = ns_;
        gamma.resize(nc_*ns_, 0.0);
        delta_gamma.resize(nc_*ns_, 0.0);
}

void Data::reset() {
    std::fill(gamma.begin(), gamma.end(), 0.0f);
    std::fill(delta_gamma.begin(), delta_gamma.end(), 0.0f);
    cl = 0.0f;
    cd = 0.0f;
    cm = 0.0f;
}

void Data::compute_freestream(f32 alpha) {
    f32 alpha_rad = alpha * PI_f / 180.0f;
    u_inf.x = std::cos(alpha_rad);
    u_inf.y = 0.0f;
    u_inf.z = std::sin(alpha_rad);

    lift_axis.x = - std::sin(alpha_rad);
    lift_axis.y = 0.0f;
    lift_axis.z = std::cos(alpha_rad);
}

void Data::postprocess() {
    for (u32 j = 0; j < ns; j++) {
        delta_gamma[j] = gamma[j];
    }
    // note: this is efficient as the memory is contiguous
    for (u32 i = 1; i < nc; i++) {
        for (u32 j = 0; j < ns; j++) {
            delta_gamma[i*ns + j] = gamma[i*ns + j] - gamma[(i-1)*ns + j];
        }
    }
}

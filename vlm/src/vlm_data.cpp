#include "vlm_data.hpp"

using namespace vlm;

Data::Data(tiny::Config& cfg) {
    s_ref = cfg().section("solver").get<f32>("s_ref", 3.25f);
    c_ref = cfg().section("solver").get<f32>("c_ref", 0.85f);
    sigma_vatistas = cfg().section("solver").get<f32>("sigma_vatistas", 0.0f);
    std::vector<f32> ref_pt_vec = cfg().section("solver").get_vector<f32>("ref_pt", {0.25f, 0.0f, 0.0f});
    ref_pt.x = ref_pt_vec[0];
    ref_pt.y = ref_pt_vec[1];
    ref_pt.z = ref_pt_vec[2];
};

void Data::alloc(const u32 size) {
    gamma.resize(size, 0.0);
    delta_gamma.resize(size, 0.0);
}

void Data::reset() {
    std::fill(gamma.begin(), gamma.end(), 0.0f);
    std::fill(delta_gamma.begin(), delta_gamma.end(), 0.0f);
    cl = 0.0f;
    cd = 0.0f;
    cm_x = 0.0f;
    cm_y = 0.0f;
    cm_z = 0.0f;
}

// alpha in radians
void Data::compute_freestream(f32 alpha_rad) {
    u_inf.x = std::cos(alpha_rad);
    u_inf.y = 0.0f;
    u_inf.z = std::sin(alpha_rad);

    lift_axis.x = - std::sin(alpha_rad);
    lift_axis.y = 0.0f;
    lift_axis.z = std::cos(alpha_rad);
}


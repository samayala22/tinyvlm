#include "vlm_data.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <cmath>

using namespace vlm;

Data::Data(tiny::Config& cfg) {
    s_ref = cfg().section("solver").get<f32>("s_ref", 0.0f);
    c_ref = cfg().section("solver").get<f32>("c_ref", 0.0f);
    sigma_vatistas = cfg().section("solver").get<f32>("sigma_vatistas", 0.0f);
    std::vector<f32> ref_pt_vec = cfg().section("solver").get_vector<f32>("ref_pt", {0.25f, 0.0f, 0.0f});
    ref_pt.x() = ref_pt_vec[0];
    ref_pt.y() = ref_pt_vec[1];
    ref_pt.z() = ref_pt_vec[2];
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
    cm.setZero();
}

Eigen::Vector3f Data::freestream(f32 alpha, f32 beta) const {
    return Eigen::Vector3f{
        u_inf * std::cos(alpha) * std::cos(beta),
        - u_inf * std::cos(alpha) * std::sin(beta),
        u_inf * std::sin(alpha),
    };
}

Eigen::Vector3f Data::lift_axis(const Eigen::Vector3f& freestream_) const {
    return (freestream_.cross(Eigen::Vector3f::UnitY()).normalized());
}

Eigen::Vector3f Data::stream_axis(const Eigen::Vector3f& freestream_) const {
    return (freestream_.normalized());
}

Eigen::Vector3f Data::freestream() const {
    return freestream(alpha, beta);
}

Eigen::Vector3f Data::lift_axis() const {
    return lift_axis(freestream());
}

Eigen::Vector3f Data::stream_axis() const {
    return stream_axis(freestream());
}




///////// 

FlowData::FlowData(const f32 alpha_, const f32 beta_, const f32 u_inf_, const f32 rho_): 
    alpha(alpha_), beta(beta_), u_inf(u_inf_), rho(rho_),
    freestream(f_freestream(alpha, beta)),
    lift_axis(f_lift_axis(freestream)),
    stream_axis(f_stream_axis(freestream)) {
}

Eigen::Vector3f FlowData::f_freestream(f32 alpha, f32 beta) const {
    return Eigen::Vector3f{
        u_inf * std::cos(alpha) * std::cos(beta),
        - u_inf * std::cos(alpha) * std::sin(beta),
        u_inf * std::sin(alpha),
    };
}

Eigen::Vector3f FlowData::f_lift_axis(const Eigen::Vector3f& freestream_) const {
    return (freestream_.cross(Eigen::Vector3f::UnitY()).normalized());
}

Eigen::Vector3f FlowData::f_stream_axis(const Eigen::Vector3f& freestream_) const {
    return (freestream_.normalized());
}

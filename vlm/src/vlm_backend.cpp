#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"

using namespace vlm;

f32 Backend::compute_coefficient_cl(const FlowData& flow) {
    return compute_coefficient_cl(flow, mesh.s_ref, 0, mesh.ns);
}

Eigen::Vector3f Backend::compute_coefficient_cm(const FlowData& flow) {
    return compute_coefficient_cm(flow, mesh.s_ref, mesh.c_ref, 0, mesh.ns);
}

f32 Backend::compute_coefficient_cd(const FlowData& flow) {
    return compute_coefficient_cd(flow, mesh.s_ref, 0, mesh.ns);
}
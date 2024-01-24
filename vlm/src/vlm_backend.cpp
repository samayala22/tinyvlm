#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"

using namespace vlm;

// Overload for the whole mesh body
f32 Backend::compute_coefficient_cl(const Mesh& mesh, const Data& data, const f32 area) {
    return compute_coefficient_cl(mesh, data, area, data.freestream(data.alpha, data.beta), 0, mesh.ns);
}

// Overload for the whole mesh body
Eigen::Vector3f Backend::compute_coefficient_cm(const Mesh& mesh, const Data& data, const f32 area, const f32 chord) {
    return compute_coefficient_cm(mesh, data, area, chord, 0, mesh.ns);
}

// Overload for the whole mesh body
f32 Backend::compute_coefficient_cd(const Mesh& mesh, const Data& data, const f32 area) {
    return compute_coefficient_cd(mesh, data, area, 0, mesh.ns);
}

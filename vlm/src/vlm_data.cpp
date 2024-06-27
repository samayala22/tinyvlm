#include "vlm_data.hpp"
#include "vlm_utils.hpp"

#include <cmath>

using namespace vlm;

FlowData::FlowData(const f32 alpha_, const f32 beta_, const f32 u_inf_, const f32 rho_): 
    alpha(alpha_), beta(beta_), u_inf(u_inf_), rho(rho_),
    freestream(compute_freestream(u_inf, alpha, beta)),
    lift_axis(compute_lift_axis(freestream)),
    stream_axis(compute_stream_axis(freestream)) {
}

FlowData::FlowData(const linalg::alias::float3& freestream_, const f32 rho_): 
    alpha(std::atan(freestream_.z / std::sqrt(freestream_.x*freestream_.x + freestream_.y*freestream_.y))),
    beta(std::atan(freestream_.y / freestream_.x)),
    u_inf(linalg::length(freestream_)),
    rho(rho_),
    freestream(freestream_),
    lift_axis(compute_lift_axis(freestream)),
    stream_axis(compute_stream_axis(freestream)) {
}

void vlm::data_alloc(const malloc_f malloc, Data* data, u64 nc, u64 ns, u64 nw) {
    const u64 nb_panels_wing = nc * ns;
    const u64 nb_panels_total = (nc+nw) * ns;
    const u64 nb_vertices_wing = (nc+1) * (ns+1);
    const u64 nb_vertices_total = (nc+nw+1) * (ns+1);
    data->lhs = (f32*)malloc(nb_panels_wing * nb_panels_wing * sizeof(f32));
    data->rhs = (f32*)malloc(nb_panels_wing * sizeof(f32));
    data->gamma = (f32*)malloc(nb_panels_total * sizeof(f32));
    data->gamma_prev = (f32*)malloc(nb_panels_wing * sizeof(f32));
    data->delta_gamma = (f32*)malloc(nb_panels_wing * sizeof(f32));
    data->rollup_vertices = (f32*)malloc(nb_vertices_total * 3 * sizeof(f32)); // TODO: this can be reduced to (nw+1)*(ns+1)*3
    data->local_velocities = (f32*)malloc(nb_panels_wing * 3 * sizeof(f32));
    data->trefftz_buffer = (f32*)malloc(ns * sizeof(f32));
}

void vlm::data_free(const free_f free, Data* data) {
    free(data->lhs);
    free(data->rhs);
    free(data->gamma);
    free(data->gamma_prev);
    free(data->delta_gamma);
    free(data->rollup_vertices);
    free(data->local_velocities);
    free(data->trefftz_buffer);    
}

#include "vlm_backend_cpu.hpp"
#include "vlm_backend_cpu_kernels.hpp"
#include "vlm_backend_cpu_kernels_ispc.h"

#include "linalg.h"
#include "tinytimer.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_executor.hpp" // includes taskflow/taskflow.hpp

#include <algorithm> // std::fill
#include <cstdio> // std::printf

#include <stdint.h>
#include <taskflow/algorithm/for_each.hpp>

#include <lapacke.h>
#include <cblas.h>

using namespace vlm;

template<typename T>
void print_buffer(const T* start, u64 size) {
    std::cout << "[";
    for (u64 i = 0; i < size; i++) {
        std::cout << start[i] << ",";
    }
    std::cout << "]\n";
}

using namespace linalg::ostream_overloads;

BackendCPU::~BackendCPU() = default; // Destructor definition

// TODO: replace any kind of size arithmetic with methods
BackendCPU::BackendCPU(Mesh& mesh) : Backend(mesh) {
    lhs.resize(mesh.nb_panels_wing() * mesh.nb_panels_wing());
    wake_buffer.resize(mesh.nb_panels_wing() * mesh.ns);

    rollup_vertices.resize(mesh.nb_vertices_total());
    uw.resize(mesh.nb_panels_wing() * mesh.ns);
    vw.resize(mesh.nb_panels_wing() * mesh.ns);
    ww.resize(mesh.nb_panels_wing() * mesh.ns);

    panel_uw.resize(mesh.nb_panels_wing());
    panel_vw.resize(mesh.nb_panels_wing());
    panel_ww.resize(mesh.nb_panels_wing());

    rhs.resize(mesh.nb_panels_wing());
    ipiv.resize(mesh.nb_panels_wing());
    gamma.resize((mesh.nc + mesh.nw) * mesh.ns); // store wake gamma as well
    gamma_prev.resize(mesh.nb_panels_wing());
    delta_gamma.resize(mesh.nb_panels_wing());
    trefftz_buffer.resize(mesh.ns);
}

void BackendCPU::reset() {
    // TODO: verify that these are needed
    std::fill(gamma.begin(), gamma.end(), 0.0f);
    std::fill(lhs.begin(), lhs.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);
}

void BackendCPU::compute_delta_gamma() {
    std::copy(gamma.data(), gamma.data()+mesh.ns, delta_gamma.data());

    // note: this is efficient as the memory is contiguous
    for (u64 i = 1; i < mesh.nc; i++) {
        for (u64 j = 0; j < mesh.ns; j++) {
            delta_gamma[i*mesh.ns + j] = gamma[i*mesh.ns + j] - gamma[(i-1)*mesh.ns + j];
        }
    }
}

void BackendCPU::compute_lhs() {
    tiny::ScopedTimer timer("LHS");
    Mesh& m = mesh;

    ispc::MeshProxy mesh_proxy = {
        m.ns, m.nc, m.nb_panels_wing(),
        {m.v.x.data(), m.v.y.data(), m.v.z.data()}, 
        {m.colloc.x.data(), m.colloc.y.data(), m.colloc.z.data()},
        {m.normal.x.data(), m.normal.y.data(), m.normal.z.data()}
    };

    const u64 zero = 0;
    const u64 end_wing = (m.nc - 1) * m.ns;

    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto wing_pass = taskflow.for_each_index(zero, end_wing, [&] (u64 i) {
        ispc::kernel_influence(mesh_proxy, lhs.data(), i, i, sigma_vatistas);
    });

    u64 idx = m.nc - 1;
    auto cond = taskflow.emplace([&]{
        return idx < m.nc + m.current_nw ? 0 : 1; // 0 means continue, 1 means break
    });
    auto wake_pass = taskflow.for_each_index(zero, m.ns, [&] (u64 j) {
        const u64 ia = (m.nc - 1) * m.ns + j;
        const u64 lidx = idx * m.ns + j;
        ispc::kernel_influence(mesh_proxy, lhs.data(), ia, lidx, sigma_vatistas);
    });
    auto back = taskflow.emplace([&]{
        idx++;
        return 0; // 0 means continue
    });
    auto sync = taskflow.placeholder();

    init.precede(wing_pass, cond);
    wing_pass.precede(sync);
    cond.precede(wake_pass, sync); // 0 and 1
    wake_pass.precede(back);
    back.precede(cond);

    Executor::get().run(taskflow).wait();
}

void kernel_cpu_rhs(u64 n, const float normal_x[], const float normal_y[], const float normal_z[], float freestream_x, float freestream_y, float freestream_z, float rhs[]) {
    for (u64 i = 0; i < n; i++) {
        rhs[i] = - (freestream_x * normal_x[i] + freestream_y * normal_y[i] + freestream_z * normal_z[i]);
    }
}

void BackendCPU::compute_rhs(const FlowData& flow) {
    const tiny::ScopedTimer timer("RHS");
    const Mesh& m = mesh;
    
    kernel_cpu_rhs(m.nb_panels_wing(), m.normal.x.data(), m.normal.y.data(), m.normal.z.data(), flow.freestream.x, flow.freestream.y, flow.freestream.z, rhs.data());
}

void BackendCPU::add_wake_influence() {
    const tiny::ScopedTimer timer("Wake Influence");

    tf::Taskflow taskflow;

    Mesh& m = mesh;

    ispc::MeshView mesh_view = {
        mesh.nc, mesh.ns, mesh.nw, mesh.current_nw,
        {mesh.v.x.data(), mesh.v.y.data(), mesh.v.z.data()},
        {mesh.colloc.x.data(), mesh.colloc.y.data(), mesh.colloc.z.data()},
        {mesh.normal.x.data(), mesh.normal.y.data(), mesh.normal.z.data()},
    };

    auto init = taskflow.placeholder();
    auto wake_influence = taskflow.for_each_index((u64)0, m.ns * m.nc, [&] (u64 lidx) {
        ispc::kernel_wake_influence(mesh_view, lidx, gamma.data(), rhs.data(), sigma_vatistas);
    });
    auto sync = taskflow.placeholder();

    init.precede(wake_influence);
    wake_influence.precede(sync);

    Executor::get().run(taskflow).wait();
}

void BackendCPU::wake_rollup(float dt) {
    const tiny::ScopedTimer timer("Wake Rollup");

    ispc::MeshView mesh_view = {
        mesh.nc, mesh.ns, mesh.nw, mesh.current_nw,
        {mesh.v.x.data(), mesh.v.y.data(), mesh.v.z.data()},
        {mesh.colloc.x.data(), mesh.colloc.y.data(), mesh.colloc.z.data()},
        {mesh.normal.x.data(), mesh.normal.y.data(), mesh.normal.z.data()},
    };

    const u64 wake_vertices_begin = (mesh.nc + mesh.nw - mesh.current_nw + 1) * (mesh.ns+1);
    const u64 wake_vertices_end = (mesh.nc + mesh.nw + 1) * (mesh.ns + 1);

    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto rollup = taskflow.for_each_index(wake_vertices_begin, wake_vertices_end, [&] (u64 vidx) {
        ispc::kernel_rollup(mesh_view, dt, rollup_vertices.x.data(), rollup_vertices.y.data(), rollup_vertices.z.data(), vidx, gamma.data(), sigma_vatistas);
    });
    auto copy = taskflow.emplace([&]{
        std::copy(rollup_vertices.x.data() + wake_vertices_begin, rollup_vertices.x.data() + wake_vertices_end, mesh.v.x.data() + wake_vertices_begin);
        std::copy(rollup_vertices.y.data() + wake_vertices_begin, rollup_vertices.y.data() + wake_vertices_end, mesh.v.y.data() + wake_vertices_begin);
        std::copy(rollup_vertices.z.data() + wake_vertices_begin, rollup_vertices.z.data() + wake_vertices_end, mesh.v.z.data() + wake_vertices_begin);
    });
    auto sync = taskflow.placeholder();
    init.precede(rollup);
    rollup.precede(copy);
    copy.precede(sync);

    Executor::get().run(taskflow).wait();

    // for (u64 vidx = wake_vertices_begin; vidx < wake_vertices_end; vidx++) {
    //     ispc::kernel_induced_vel(mesh_view, dt, rollup_vertices.x.data(), rollup_vertices.y.data(), rollup_vertices.z.data(), vidx, gamma.data(), sigma_vatistas);
    // }
    // std::copy(rollup_vertices.x.data() + wake_vertices_begin, rollup_vertices.x.data() + rollup_vertices.size, mesh.v.x.data() + wake_vertices_begin);
    // std::copy(rollup_vertices.y.data() + wake_vertices_begin, rollup_vertices.y.data() + rollup_vertices.size, mesh.v.y.data() + wake_vertices_begin);
    // std::copy(rollup_vertices.z.data() + wake_vertices_begin, rollup_vertices.z.data() + rollup_vertices.size, mesh.v.z.data() + wake_vertices_begin);
}

void BackendCPU::shed_gamma() {
    const Mesh& m = mesh;
    const u64 wake_row_start = (m.nc + m.nw - m.current_nw - 1) * m.ns;

    std::copy(gamma.data(), gamma.data() + m.nb_panels_wing(), gamma_prev.data()); // store current timestep for delta_gamma
    //std::copy(delta_gamma.data(), delta_gamma.data() + m.nb_panels_wing(), gamma_prev.data()); // store current timestep for delta_gamma
    std::copy(gamma.data() + m.ns * (m.nc-1), gamma.data() + m.nb_panels_wing(), gamma.data() + wake_row_start);
}

void BackendCPU::compute_rhs(const SoA_3D_t<f32>& velocities) {
    // const tiny::ScopedTimer timer("Rebuild RHS");
    const Mesh& m = mesh;
    for (u64 i = 0; i < m.nb_panels_wing(); i++) {
        rhs[i] = - (velocities.x[i] * m.normal.x[i] + velocities.y[i] * m.normal.y[i] + velocities.z[i] * m.normal.z[i]); 
    }
}

void BackendCPU::lu_factor() {
    const tiny::ScopedTimer timer("Factor");
    const int32_t n = static_cast<int32_t>(mesh.nb_panels_wing());
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, lhs.data(), n, ipiv.data());
}

void BackendCPU::lu_solve() {
    const tiny::ScopedTimer timer("Solve");
    const int32_t n = static_cast<int32_t>(mesh.nb_panels_wing());
    std::copy(rhs.begin(), rhs.end(), gamma.begin());

    LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, 1, lhs.data(), n, ipiv.data(), gamma.data(), n);
}

f32 BackendCPU::compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) {
    const tiny::ScopedTimer timer("Compute CL");

    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    
    f32 cl = 0.0f;

    for (u64 u = 0; u < mesh.nc; u++) {
        for (u64 v = j; v < j + n; v++) {
            const u64 li = u * mesh.ns + v; // linear index
            const linalg::alias::float3 v0 = mesh.get_v0(li);
            const linalg::alias::float3 v1 = mesh.get_v1(li);
            // const linalg::alias::float3 v3 = mesh.get_v3(li);
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = v1 - v0;
            // const linalg::alias::float3 local_left_chord = linalg::normalize(v3 - v0);
            // const linalg::alias::float3 projected_vector = linalg::dot(dl, local_left_chord) * local_left_chord;
            // dl -= projected_vector; // orthogonal leading edge vector
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * delta_gamma[li];
            cl += linalg::dot(force, flow.lift_axis); // projection on the body lift axis
        }
    }
    cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;

    return cl;
}

f32 BackendCPU::compute_coefficient_unsteady_cl(const linalg::alias::float3& freestream, const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) {
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    
    f32 cl = 0.0f;
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    const linalg::alias::float3 span_axis{mesh.frame.y[0], mesh.frame.y[1], mesh.frame.y[2]};
    const linalg::alias::float3 lift_axis = linalg::normalize(linalg::cross(freestream, span_axis));

    for (u64 u = 0; u < mesh.nc; u++) {
        for (u64 v = j; v < j + n; v++) {
            const u64 li = u * mesh.ns + v; // linear index

            linalg::alias::float3 V{vel.x[li], vel.y[li], vel.z[li]}; // local velocity (freestream + displacement vel)

            const linalg::alias::float3 v0 = mesh.get_v0(li);
            const linalg::alias::float3 v1 = mesh.get_v1(li);
            const linalg::alias::float3 v2 = mesh.get_v2(li);
            const linalg::alias::float3 v3 = mesh.get_v3(li);
            const linalg::alias::float3 normal{mesh.normal.x[li], mesh.normal.y[li], mesh.normal.z[li]};

            linalg::alias::float3 force = {0.0f, 0.0f, 0.0f};
            const f32 gamma_dt = (gamma[li] - gamma_prev[li]) / dt; // backward difference

            // Joukowski method
            force += rho * delta_gamma[li] * linalg::cross(V, v1 - v0);
            force += rho * gamma_dt * mesh.area[li] * normal;

            // Katz Plotkin method
            // linalg::alias::float3 delta_p = {0.0f, 0.0f, 0.0f};
            // const f32 delta_gamma_i = (u == 0) ? gamma[li] : gamma[li] - gamma[(u-1) * mesh.ns + v];
            // const f32 delta_gamma_j = (v == 0) ? gamma[li] : gamma[li] - gamma[u * mesh.ns + v - 1];
            // delta_p += rho * linalg::dot(freestream, linalg::normalize(v1 - v0)) * delta_gamma_j / mesh.panel_width_y(u, v);
            // delta_p += rho * linalg::dot(freestream, linalg::normalize(v3 - v0)) * delta_gamma_i / mesh.panel_length(u, v);
            // delta_p += gamma_dt;
            // force = (delta_p * mesh.area[li]) * normal;

            // force /= linalg::length2(freestream);
            
            cl += linalg::dot(force, lift_axis);
        }
    }
    cl /= 0.5f * rho * linalg::length2(freestream) * area; // TODO: remove uinf hardcoded as 1.0f

    return cl;
}

linalg::alias::float3 BackendCPU::compute_coefficient_cm(
    const FlowData& flow,
    const f32 area,
    const f32 chord,
    const u64 j,
    const u64 n)
{
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    linalg::alias::float3 cm(0.f, 0.f, 0.f);

    for (u64 u = 0; u < mesh.nc; u++) {
        for (u64 v = j; v < j + n; v++) {
            const u64 li = u * mesh.ns + v; // linear index
            const linalg::alias::float3 v0 = mesh.get_v0(li);
            const linalg::alias::float3 v1 = mesh.get_v1(li);
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = v1 - v0;
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 dst_to_ref = mesh.ref_pt - 0.5f * (v0 + v1);
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * delta_gamma[li];
            cm += linalg::cross(force, dst_to_ref);
        }
    }
    cm /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area * chord;
    return cm;
}

f32 BackendCPU::compute_coefficient_cd(
    const FlowData& flow,
    const f32 area,
    const u64 j,
    const u64 n) 
{
    tiny::ScopedTimer timer("Compute CD");
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    std::fill(trefftz_buffer.begin(), trefftz_buffer.end(), 0.0f);

    Mesh& m = mesh;
    ispc::MeshProxy mesh_proxy = {
        m.ns, m.nc, m.nb_panels_wing(),
        {m.v.x.data(), m.v.y.data(), m.v.z.data()}, 
        {m.colloc.x.data(), m.colloc.y.data(), m.colloc.z.data()},
        {m.normal.x.data(), m.normal.y.data(), m.normal.z.data()}
    };
    f32 cd = ispc::kernel_trefftz_cd(mesh_proxy, gamma.data(), trefftz_buffer.data(), j, n, sigma_vatistas);
    cd /= linalg::length2(flow.freestream) * area;
    return cd;
}

// Using Trefftz plane
// f32 BackendCPU::compute_coefficient_cl(
//     const FlowData& flow,
//     const f32 area,
//     const u64 j,
//     const u64 n) 
// {
//     Mesh& m = mesh;
//     ispc::MeshProxy mesh_proxy = {
//         m.ns, m.nc, m.nb_panels_wing(),
//         {m.v.x.data(), m.v.y.data(), m.v.z.data()}, 
//         {m.colloc.x.data(), m.colloc.y.data(), m.colloc.z.data()},
//         {m.normal.x.data(), m.normal.y.data(), m.normal.z.data()}
//     };
//     f32 cl = ispc::kernel_trefftz_cl(mesh_proxy, gamma.data(), j, n);
//     cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;
//     return cl;
// }

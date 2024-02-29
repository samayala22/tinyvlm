#include "vlm_backend_cpu.hpp"
#include "vlm_backend_cpu_kernels.hpp"
#include "vlm_backend_cpu_kernels_ispc.h"

#include "linalg.h"
#include "tinytimer.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_utils.hpp"
#include "vlm_executor.hpp" // includes taskflow/taskflow.hpp

#include <algorithm> // std::fill
#include <cstdio> // std::printf

#include <stdint.h>
#include <taskflow/algorithm/for_each.hpp>

#include <lapacke.h>
#include <cblas.h>

using namespace vlm;

BackendCPU::~BackendCPU() = default; // Destructor definition

BackendCPU::BackendCPU(Mesh& mesh) : Backend(mesh) {
    lhs.resize(mesh.nb_panels_wing() * mesh.nb_panels_wing());
    rhs.resize(mesh.nb_panels_wing());
    ipiv.resize(mesh.nb_panels_wing());
    gamma.resize(mesh.nb_panels_wing());
    delta_gamma.resize(mesh.nb_panels_wing());
    trefftz_buffer.resize(mesh.ns);
}

void BackendCPU::reset() {
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

void BackendCPU::compute_lhs(const FlowData& flow) {
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
        ispc::kernel_influence(mesh_proxy, lhs.data(), i, i, flow.sigma_vatistas);
    });

    u64 idx = m.nc - 1;
    auto cond = taskflow.emplace([&]{
        return idx < m.nc + m.nw ? 0 : 1; // 0 means continue, 1 means break
    });
    auto wake_pass = taskflow.for_each_index(zero, m.ns, [&] (u64 j) {
        const u64 ia = (m.nc - 1) * m.ns + j;
        const u64 lidx = idx * m.ns + j;
        ispc::kernel_influence(mesh_proxy, lhs.data(), ia, lidx, flow.sigma_vatistas);
    });
    auto back = taskflow.emplace([&]{
        idx++;
        return 0; // 0 means continue
    });
    auto sync = taskflow.placeholder();

    init.precede(wing_pass, cond);
    wing_pass.precede(sync);
    cond.precede(wake_pass, sync);
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
    tiny::ScopedTimer timer("RHS");
    const Mesh& m = mesh;
    
    kernel_cpu_rhs(m.nb_panels_wing(), m.normal.x.data(), m.normal.y.data(), m.normal.z.data(), flow.freestream.x, flow.freestream.y, flow.freestream.z, rhs.data());
}

void BackendCPU::compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) {
    tiny::ScopedTimer timer("Rebuild RHS");
    assert(section_alphas.size() == mesh.ns);
    const Mesh& m = mesh;
    for (u64 i = 0; i < mesh.nc; i++) {
        for (u64 j = 0; j < mesh.ns; j++) {
            const u64 li = i * mesh.ns + j; // linear index
            const linalg::alias::float3 freestream = compute_freestream(flow.u_inf, section_alphas[j], flow.beta);
            rhs[li] = - (freestream.x * m.normal.x[li] + freestream.y * m.normal.y[li] + freestream.z * m.normal.z[li]);
        }
    }
}

void BackendCPU::lu_factor() {
    tiny::ScopedTimer timer("Factor");
    const int32_t n = static_cast<int32_t>(mesh.nb_panels_wing());
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, lhs.data(), n, ipiv.data());
}

void BackendCPU::lu_solve() {
    tiny::ScopedTimer timer("Solve");
    const int32_t n = static_cast<int32_t>(mesh.nb_panels_wing());
    std::copy(rhs.begin(), rhs.end(), gamma.begin());

    LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, 1, lhs.data(), n, ipiv.data(), gamma.data(), n);
}

f32 BackendCPU::compute_coefficient_cl(const FlowData& flow, const f32 area,
    const u64 j, const u64 n) {
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    
    f32 cl = 0.0f;

    for (u64 u = 0; u < mesh.nc; u++) {
        for (u64 v = j; v < j + n; v++) {
            const u64 li = u * mesh.ns + v; // linear index
            const linalg::alias::float3 v0 = mesh.get_v0(li);
            const linalg::alias::float3 v1 = mesh.get_v1(li);
            const linalg::alias::float3 v3 = mesh.get_v3(li);
            // Leading edge vector pointing outward from wing root
            linalg::alias::float3 dl = v1 - v0;
            const linalg::alias::float3 local_left_chord = linalg::normalize(v3 - v0);
            const linalg::alias::float3 projected_vector = linalg::dot(dl, local_left_chord) * local_left_chord;
            dl -= projected_vector; // orthogonal leading edge vector
            std::printf("projected_vector: %f %f %f\n", projected_vector.x, projected_vector.y, projected_vector.z);
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.stream_axis, dl) * flow.rho * delta_gamma[li];
            cl += linalg::dot(force, flow.lift_axis);
        }
    }
    cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;

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
    f32 cd = ispc::kernel_trefftz_cd(mesh_proxy, gamma.data(), trefftz_buffer.data(), j, n, flow.sigma_vatistas);
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

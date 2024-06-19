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
using namespace linalg::ostream_overloads;

BackendCPU::~BackendCPU() = default; // Destructor definition

BackendCPU::BackendCPU(MeshGeom* mesh, u64 timesteps) {
    allocator.h_malloc = std::malloc;
    allocator.d_malloc = std::malloc;
    allocator.h_free = std::free;
    allocator.d_free = std::free;
    allocator.hh_memcpy = std::memcpy;
    allocator.hd_memcpy = std::memcpy;
    allocator.dh_memcpy = std::memcpy;
    allocator.dd_memcpy = std::memcpy;
    allocator.h_memset = std::memset;
    allocator.d_memset = std::memset;
    init(mesh, timesteps);

    d_solver_info = (i32*)allocator.d_malloc(sizeof(i32));
    d_solver_ipiv = (i32*)allocator.d_malloc(hd_mesh->nc*hd_mesh->ns*sizeof(i32));
}

void BackendCPU::reset() {
    const u64 nb_panels_wing = hd_mesh->nc * hd_mesh->ns;

    // std::fill(gamma.begin(), gamma.end(), 0.0f);
    std::fill(hd_data->lhs, hd_data->lhs + nb_panels_wing*nb_panels_wing, 0.0f); // influence kernel is +=
    // std::fill(rhs.begin(), rhs.end(), 0.0f);
}

void BackendCPU::compute_delta_gamma() {
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nwa = hd_mesh->nwa;
    // const tiny::ScopedTimer timer("Delta Gamma");
    std::copy(dd_data->gamma, dd_data->gamma+ns, dd_data->delta_gamma);

    // note: this is efficient as the memory is contiguous
    for (u64 i = 1; i < nc; i++) {
        for (u64 j = 0; j < ns; j++) {
            dd_data->delta_gamma[i*ns + j] = dd_data->gamma[i*ns + j] - dd_data->gamma[(i-1)*ns + j];
        }
    }
}

void BackendCPU::lhs_assemble() {
    tiny::ScopedTimer timer("LHS");
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nwa = hd_mesh->nwa;

    const u64 zero = 0;
    const u64 end_wing = (nc - 1) * ns;

    ispc::Mesh2 mesh = dd_mesh;

    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto wing_pass = taskflow.for_each_index(zero, end_wing, [&] (u64 i) {
        ispc::kernel_influence(dd_mesh, dd_data->lhs, i, i, sigma_vatistas);
    });

    u64 idx = nc - 1;
    auto cond = taskflow.emplace([&]{
        return idx < nc + nwa ? 0 : 1; // 0 means continue, 1 means break
    });
    auto wake_pass = taskflow.for_each_index(zero, ns, [&] (u64 j) {
        const u64 ia = (nc - 1) * ns + j;
        const u64 lidx = idx * ns + j;
        ispc::kernel_influence(dd_mesh, dd_data->lhs, ia, lidx, sigma_vatistas);
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

void BackendCPU::compute_rhs() {
    const tiny::ScopedTimer timer("RHS");
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nb_panels_wing = nc * ns;

    for (u64 i = 0; i < nb_panels_wing; i++) {
        dd_data->rhs[i] = - (
            dd_data->local_velocities[i + 0 * nb_panels_wing] * dd_mesh->normals[i + 0 * nb_panels_wing] +
            dd_data->local_velocities[i + 1 * nb_panels_wing] * dd_mesh->normals[i + 1 * nb_panels_wing] +
            dd_data->local_velocities[i + 2 * nb_panels_wing] * dd_mesh->normals[i + 2 * nb_panels_wing]);
    }
}

void BackendCPU::add_wake_influence() {
    // const tiny::ScopedTimer timer("Wake Influence");
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto wake_influence = taskflow.for_each_index((u64)0, ns * nc, [&] (u64 lidx) {
        ispc::kernel_wake_influence(dd_mesh, lidx, dd_data->gamma, dd_data->rhs, sigma_vatistas);
    });
    auto sync = taskflow.placeholder();

    init.precede(wake_influence);
    wake_influence.precede(sync);

    Executor::get().run(taskflow).wait();
}

void BackendCPU::wake_rollup(float dt) {
    const tiny::ScopedTimer timer("Wake Rollup");

    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nwa = hd_mesh->nwa;

    const u64 wake_vertices_begin = (nc + nw - nwa + 1) * (ns+1);
    const u64 wake_vertices_end = (nc + nw + 1) * (ns + 1);

    const f32* rx = dd_data->rollup_vertices + (nc+nw+1)*(ns+1)*0;
    const f32* ry = dd_data->rollup_vertices + (nc+nw+1)*(ns+1)*1;
    const f32* rz = dd_data->rollup_vertices + (nc+nw+1)*(ns+1)*2;

    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto rollup = taskflow.for_each_index(wake_vertices_begin, wake_vertices_end, [&] (u64 vidx) {
        ispc::kernel_rollup(dd_mesh, dt, rx, ry, rz, vidx, dd_data->gamma, sigma_vatistas);
    });
    auto copy = taskflow.emplace([&]{
        std::copy(rx + wake_vertices_begin, rx + wake_vertices_end, PTR_MESH_V(dd_mesh, 0,0,0) + wake_vertices_begin);
        std::copy(ry + wake_vertices_begin, ry + wake_vertices_end, PTR_MESH_V(dd_mesh, 0,0,1) + wake_vertices_begin);
        std::copy(rz + wake_vertices_begin, rz + wake_vertices_end, PTR_MESH_V(dd_mesh, 0,0,2) + wake_vertices_begin);
    });
    auto sync = taskflow.placeholder();
    init.precede(rollup);
    rollup.precede(copy);
    copy.precede(sync);

    Executor::get().run(taskflow).wait();
}

void BackendCPU::shed_gamma() {
    // const tiny::ScopedTimer timer("Shed Gamma");
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nwa = hd_mesh->nwa;
    const u64 nb_panels_wing = nc * ns;
    const u64 wake_row_start = (nc + nw - nwa - 1) * ns;

    std::copy(dd_data->gamma, dd_data->gamma + nb_panels_wing, dd_data->gamma_prev); // store current timestep for delta_gamma
    std::copy(dd_data->gamma + ns * (nc-1), dd_data->gamma + nb_panels_wing, dd_data->gamma + wake_row_start);
}

void BackendCPU::lu_factor() {
    const tiny::ScopedTimer timer("Factor");
    const u64 nb_panels_wing = hd_mesh->nc * hd_mesh->ns;
    const int32_t n = static_cast<int32_t>(nb_panels_wing);
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, dd_data->lhs, n, d_solver_ipiv);
}

void BackendCPU::lu_solve() {
    // const tiny::ScopedTimer timer("Solve");
    const u64 nb_panels_wing = hd_mesh->nc * hd_mesh->ns;
    const int32_t n = static_cast<int32_t>(nb_panels_wing);
    std::copy(dd_data->rhs, dd_data->rhs+nb_panels_wing, dd_data->gamma);

    LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, 1, dd_data->lhs, n, d_solver_ipiv, dd_data->gamma, n);
}

f32 BackendCPU::compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) {
    const tiny::ScopedTimer timer("Compute CL");

    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nwa = hd_mesh->nwa;
    const u64 nb_panels_wing = nc * ns;

    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    
    f32 cl = 0.0f;

    for (u64 u = 0; u < nc; u++) {
        for (u64 v = j; v < j + n; v++) {
            const linalg::alias::float3 v0{*PTR_MESH_V(dd_mesh, u, v, 0), *PTR_MESH_V(dd_mesh, u, v, 1), *PTR_MESH_V(dd_mesh, u, v, 2)}; // upper left
            const linalg::alias::float3 v1{*PTR_MESH_V(dd_mesh, u, v+1, 0), *PTR_MESH_V(dd_mesh, u, v+1, 1), *PTR_MESH_V(dd_mesh, u, v+1, 2)}; // upper right
            // const linalg::alias::float3 v3 = mesh.get_v3(li);
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = v1 - v0;
            // const linalg::alias::float3 local_left_chord = linalg::normalize(v3 - v0);
            // const linalg::alias::float3 projected_vector = linalg::dot(dl, local_left_chord) * local_left_chord;
            // dl -= projected_vector; // orthogonal leading edge vector
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * dd_data->delta_gamma[u * ns + v];
            cl += linalg::dot(force, flow.lift_axis); // projection on the body lift axis
        }
    }
    cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;

    return cl;
}

f32 BackendCPU::compute_coefficient_unsteady_cl(const linalg::alias::float3& freestream, const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) {
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);

    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nwa = hd_mesh->nwa;
    const u64 nb_panels_wing = nc * ns;
    
    f32 cl = 0.0f;
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    const linalg::alias::float3 span_axis{dd_mesh->frame + 4};
    const linalg::alias::float3 lift_axis = linalg::normalize(linalg::cross(freestream, span_axis));

    for (u64 u = 0; u < nc; u++) {
        for (u64 v = j; v < j + n; v++) {
            const u64 li = u * ns + v; // linear index

            linalg::alias::float3 V{vel.x[li], vel.y[li], vel.z[li]}; // local velocity (freestream + displacement vel)

            const linalg::alias::float3 v0{*PTR_MESH_V(dd_mesh, u, v, 0), *PTR_MESH_V(dd_mesh, u, v, 1), *PTR_MESH_V(dd_mesh, u, v, 2)}; // upper left
            const linalg::alias::float3 v1{*PTR_MESH_V(dd_mesh, u, v+1, 0), *PTR_MESH_V(dd_mesh, u, v+1, 1), *PTR_MESH_V(dd_mesh, u, v+1, 2)}; // upper right
            const linalg::alias::float3 normal{*PTR_MESH_N(dd_mesh, u, v, 0), *PTR_MESH_N(dd_mesh, u, v, 1), *PTR_MESH_N(dd_mesh, u, v, 2)}; // normal

            linalg::alias::float3 force = {0.0f, 0.0f, 0.0f};
            const f32 gamma_dt = (dd_data->gamma[li] - dd_data->gamma_prev[li]) / dt; // backward difference

            // Joukowski method
            force += rho * dd_data->delta_gamma[li] * linalg::cross(V, v1 - v0);
            force += rho * gamma_dt * dd_mesh->area[li] * normal;

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
    cl /= 0.5f * rho * linalg::length2(freestream) * area;

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

    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nwa = hd_mesh->nwa;
    const u64 nb_panels_wing = nc * ns;

    linalg::alias::float3 cm(0.f, 0.f, 0.f);
    const linalg::alias::float3 ref_pt{dd_mesh->frame + 12}; // frame origin as moment pt

    for (u64 u = 0; u < nc; u++) {
        for (u64 v = j; v < j + n; v++) {
            const u64 li = u * ns + v; // linear index
            const linalg::alias::float3 v0{*PTR_MESH_V(dd_mesh, u, v, 0), *PTR_MESH_V(dd_mesh, u, v, 1), *PTR_MESH_V(dd_mesh, u, v, 2)}; // upper left
            const linalg::alias::float3 v1{*PTR_MESH_V(dd_mesh, u, v+1, 0), *PTR_MESH_V(dd_mesh, u, v+1, 1), *PTR_MESH_V(dd_mesh, u, v+1, 2)}; // upper right
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = v1 - v0;
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 dst_to_ref = ref_pt - 0.5f * (v0 + v1);
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * dd_data->delta_gamma[li];
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
    std::fill(dd_data->trefftz_buffer, dd_data->trefftz_buffer + hd_mesh->ns, 0.0f);

    f32 cd = ispc::kernel_trefftz_cd(dd_mesh, dd_data->gamma, dd_data->trefftz_buffer, j, n, sigma_vatistas);
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

void BackendCPU::set_velocities(const linalg::alias::float3& vel) {
    const u64 nb_panels_wing = hd_mesh->nc * hd_mesh->ns;
    std::fill(dd_data->local_velocities[0 * nb_panels_wing], dd_data->local_velocities[1 * nb_panels_wing], vel.x);
    std::fill(dd_data->local_velocities[1 * nb_panels_wing], dd_data->local_velocities[2 * nb_panels_wing], vel.y);
    std::fill(dd_data->local_velocities[2 * nb_panels_wing], dd_data->local_velocities[3 * nb_panels_wing], vel.z);
}

void BackendCPU::set_velocities(const SoA_3D_t<f32>& vels) {
    const u64 nb_panels_wing = hd_mesh->nc * hd_mesh->ns;
    std::copy(vels.x.begin(), vels.x.end(), dd_data->local_velocities[0 * nb_panels_wing]);
    std::copy(vels.y.begin(), vels.y.end(), dd_data->local_velocities[1 * nb_panels_wing]);
    std::copy(vels.z.begin(), vels.z.end(), dd_data->local_velocities[2 * nb_panels_wing]);
}

void BackendCPU::mesh_metrics() {

}

/// @brief Computes the mean chord of a set of panels
/// @details
/// Mean Aerodynamic Chord = \frac{2}{S} \int_{0}^{b/2} c(y)^{2} dy
/// Integration using the Trapezoidal Rule
/// Validated against empirical formulas for tapered wings
/// @param j first panel index spanwise
/// @param n number of panels spanwise
/// @return mean chord of the set of panels
f32 BackendCPU::mesh_mac(u64 j, u64 n) {
    assert(j + n <= hd_mesh->ns); // spanwise range
    assert(n > 0);
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;

    // Leading edge vertex
    f32* lvx = PTR_MESH_V(hd_mesh, 0,0,0);
    f32* lvy = PTR_MESH_V(hd_mesh, 0,0,1);
    f32* lvz = PTR_MESH_V(hd_mesh, 0,0,2);
    // Trailing edge vertex
    f32* tvx = PTR_MESH_V(hd_mesh, nc,0,0);
    f32* tvy = PTR_MESH_V(hd_mesh, nc,0,1);
    f32* tvz = PTR_MESH_V(hd_mesh, nc,0,2);

    f32 mac = 0.0f;
    // loop over panel chordwise sections in spanwise direction
    // Note: can be done optimally with vertical fused simd
    for (u64 v = j; v < j+n; v++) {
        // left and right chord lengths
        const f32 dx0 = tvx[v] - lvx[v];
        const f32 dy0 = tvy[v] - lvy[v];
        const f32 dz0 = tvz[v] - lvz[v];
        const f32 dx1 = tvx[v + 1] - lvx[v + 1];
        const f32 dy1 = tvy[v + 1] - lvy[v + 1];
        const f32 dz1 = tvz[v + 1] - lvz[v + 1];
        const f32 c0 = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
        const f32 c1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
        // Panel width
        const f32 dx3 = lvx[v + 1] - lvx[v];
        const f32 dy3 = lvy[v + 1] - lvy[v];
        const f32 dz3 = lvz[v + 1] - lvz[v];
        const f32 width = std::sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

        mac += 0.5f * (c0 * c0 + c1 * c1) * width;
    }
    // Since we divide by half the total wing area (both sides) we dont need to multiply by 2
    return mac / mesh_area(0, j, nc, n);
}

f32 BackendCPU::mesh_area(const u64 i, const u64 j, const u64 m, const u64 n) {
    assert(i + m <= nc);
    assert(j + n <= ns);
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;

    const f32* areas = hd_mesh->area + j + i * (ns);
    f32 area_sum = 0.0f;
    for (u64 u = 0; u < m; u++) {
        for (u64 v = 0; v < n; v++) {
            area_sum += areas[v + u * ns];
        }
    }
    return area_sum;
}

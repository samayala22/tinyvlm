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
}

void BackendCPU::reset() {
    // TODO: verify that these are needed
    // std::fill(gamma.begin(), gamma.end(), 0.0f);
    // std::fill(lhs.begin(), lhs.end(), 0.0f);
    // std::fill(rhs.begin(), rhs.end(), 0.0f);
}

void BackendCPU::compute_delta_gamma() {
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
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

void BackendCPU::compute_rhs() {
    const tiny::ScopedTimer timer("RHS");
    const u64 nc = hd_mesh->nc;
    const u64 ns = hd_mesh->ns;
    const u64 nw = hd_mesh->nw;
    const u64 nb_panels_wing = nc * ns;

    for (u64 i = 0; i < nb_panels_wing(); i++) {
        rhs[i] = - (
            dd_data->local_velocities[i + 0 * nb_panels_wing] * dd_mesh->normal[i + 0 * nb_panels_wing] +
            dd_data->local_velocities[i + 1 * nb_panels_wing] * dd_mesh->normal[i + 1 * nb_panels_wing] +
            dd_data->local_velocities[i + 2 * nb_panels_wing] * dd_mesh->normal[i + 2 * nb_panels_wing]);
    }
}

void BackendCPU::add_wake_influence() {
    // const tiny::ScopedTimer timer("Wake Influence");

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
}

void BackendCPU::shed_gamma() {
    // const tiny::ScopedTimer timer("Shed Gamma");
    const Mesh& m = mesh;
    const u64 wake_row_start = (m.nc + m.nw - m.current_nw - 1) * m.ns;

    std::copy(gamma.data(), gamma.data() + m.nb_panels_wing(), gamma_prev.data()); // store current timestep for delta_gamma
    //std::copy(delta_gamma.data(), delta_gamma.data() + m.nb_panels_wing(), gamma_prev.data()); // store current timestep for delta_gamma
    std::copy(gamma.data() + m.ns * (m.nc-1), gamma.data() + m.nb_panels_wing(), gamma.data() + wake_row_start);
}

void BackendCPU::lu_factor() {
    const tiny::ScopedTimer timer("Factor");
    const int32_t n = static_cast<int32_t>(mesh.nb_panels_wing());
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, lhs.data(), n, ipiv.data());
}

void BackendCPU::lu_solve() {
    // const tiny::ScopedTimer timer("Solve");
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

void BackendCPU::set_velocities(const linalg::alias::float3& vel) {
    std::fill(local_velocities.x.begin(), local_velocities.x.end(), vel.x);
    std::fill(local_velocities.y.begin(), local_velocities.y.end(), vel.y);
    std::fill(local_velocities.z.begin(), local_velocities.z.end(), vel.z);
}

void BackendCPU::set_velocities(const SoA_3D_t<f32>& vels) {
    std::copy(vels.x.begin(), vels.x.end(), local_velocities.x.begin());
    std::copy(vels.y.begin(), vels.y.end(), local_velocities.y.begin());
    std::copy(vels.z.begin(), vels.z.end(), local_velocities.z.begin());
}

/// @brief Computes the chord length of a chordwise segment
/// @details Since the mesh follows the camber line, the chord length is computed
/// as the distance between the first and last vertex of a chordwise segment
/// @param j chordwise segment index
f32 Mesh::chord_length(const u64 j) const {
    const f32 dx = v.x[j + nc * (ns+1)] - v.x[j];
    const f32 dy = 0.0f; // chordwise segments are parallel to the x axis
    const f32 dz = v.z[j + nc * (ns+1)] - v.z[j];

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// i: chordwise index, j: spanwise index, k: x,y,z dim
#define PTR_MESH_V(m, i,j,k) (m->vertices + j + i * (m->ns+1) + k * (m->nc+m->nw+1) * (m->ns+1))

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
    u64 nc = hd_mesh->nc;
    u64 ns = hd_mesh->ns;
    u64 nw = hd_mesh->nw;

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
    u64 nc = hd_mesh->nc;
    u64 ns = hd_mesh->ns;
    u64 nw = hd_mesh->nw;

    const f32* areas = hd_mesh->area + j + i * (m->ns);
    f32 area_sum = 0.0f;
    for (u64 u = 0; u < m; u++) {
        for (u64 v = 0; v < n; v++) {
            area_sum += areas[v + u * ns];
        }
    }
    return area_sum;
}

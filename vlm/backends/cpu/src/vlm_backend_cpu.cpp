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
#include <iostream> // std::cout
#include <cstdio> // std::printf

#include <stdint.h>
#include <taskflow/algorithm/for_each.hpp>

#include <lapacke.h>
#include <cblas.h>

using namespace vlm;
using namespace linalg::ostream_overloads;

class MemoryCPU final : public Memory {
    public:
        MemoryCPU() : Memory(true) {}
        ~MemoryCPU() = default;
        void* alloc(MemoryLocation location, std::size_t size) const override {return std::malloc(size);}
        void free(MemoryLocation location, void* ptr) const override {std::free(ptr);}
        void copy(MemoryTransfer transfer, void* dst, const void* src, std::size_t size) const override {std::memcpy(dst, src, size);}
        void fill_f32(MemoryLocation location, float* ptr, float value, std::size_t size) const override {std::fill(ptr, ptr + size, value);}
};

BackendCPU::BackendCPU() : Backend(std::make_unique<MemoryCPU>()) {}
BackendCPU::~BackendCPU() = default;

// void BackendCPU::reset() {
//     const u64 nb_panels_wing = hd_mesh->nc * hd_mesh->ns;
//     const u64 nb_vertices_wing = (hd_mesh->nc+1)*(hd_mesh->ns+1);


//     // std::fill(gamma.begin(), gamma.end(), 0.0f);
//     std::fill(hd_data->lhs, hd_data->lhs + nb_panels_wing*nb_panels_wing, 0.0f); // influence kernel is +=
//     // std::fill(rhs.begin(), rhs.end(), 0.0f);
//     memory->dd_copy(PTR_MESH_V(hd_mesh, 0, 0, 0), PTR_MESHGEOM_V(hd_mesh_geom, 0, 0, 0), nb_vertices_wing*sizeof(f32));
//     memory->dd_copy(PTR_MESH_V(hd_mesh, 0, 0, 1), PTR_MESHGEOM_V(hd_mesh_geom, 0, 0, 1), nb_vertices_wing*sizeof(f32));
//     memory->dd_copy(PTR_MESH_V(hd_mesh, 0, 0, 2), PTR_MESHGEOM_V(hd_mesh_geom, 0, 0, 2), nb_vertices_wing*sizeof(f32));

//     dd_mesh->nwa = 0;
// }

void BackendCPU::gamma_delta(f32* gamma_delta, const f32* gamma) {
    tf::Taskflow graph;

    auto begin = graph.placeholder();
    auto end = graph.placeholder();

    for (const auto& p : prm)  {
        f32* p_gamma_delta = gamma_delta + p.off_wing_p;
        const f32* p_gamma = gamma + p.off_wing_p;
        tf::Task first_row = graph.emplace([&]{
            memory->copy(MemoryTransfer::DeviceToDevice, p_gamma_delta, p_gamma, p.ns * sizeof(f32));
        });
        tf::Task remaining_rows = graph.for_each_index((u64)1,p.nc, [&] (u64 b, u64 e) {
            for (u64 i = b; i < e; i++) {
                for (u64 j = 0; j < p.ns; j++) {
                    p_gamma_delta[i*p.ns + j] = p_gamma[i*p.ns + j] - p_gamma[(i-1)*p.ns + j];
                }
            }
        });

        begin.precede(first_row);
        begin.precede(remaining_rows);
        first_row.precede(end);
        remaining_rows.precede(end);
    };

    Executor::get().run(graph).wait();
    graph.dump(std::cout); 

    // std::copy(gamma,gamma+ns, delta_gamma);

    // // note: this is efficient as the memory is contiguous
    // for (u64 i = 1; i < nc; i++) {
    //     for (u64 j = 0; j < ns; j++) {
    //         delta_gamma[i*ns + j] = gamma[i*ns + j] - gamma[(i-1)*ns + j];
    //     }
    // }
}

void kernel_inf(u64 m, f32* lhs, f32* influenced, u64 influenced_ld, f32* influencer, u64 influencer_ld, u64 influencer_rld, f32* normals, u64 normals_ld);

void BackendCPU::lhs_assemble(f32* lhs, const f32* colloc, const f32* normals, const f32* verts_wing, const f32* verts_wake) {
    // tiny::ScopedTimer timer("LHS");

    const u64 total_nb_panels_wing = prm.back().off_wing_p + prm.back().nb_panels_wing();
    const u64 total_nb_vertices_wing = prm.back().off_wing_v + prm.back().nb_vertices_wing();
    
    tf::Taskflow graph;

    auto begin = graph.placeholder();
    auto end = graph.placeholder();

    for (const auto& m_col : prm)  {
        f32* lhs_section = lhs + m_col.off_wing_p * total_nb_panels_wing;
        const f32* vwing_section = verts_wing + m_col.off_wing_v;
        const f32* vwake_section = verts_wake + m_col.off_wake_v;
        const u64 zero = 0;
        const u64 end_wing = (m_col.nc - 1) * m_col.ns;
        const u64 m = total_nb_panels_wing; // sections span the whole rows of lhs matrix
        u64 w = 0; // wake row index (0 to nwa)

        auto wing_pass = graph.for_each_index(zero, end_wing, [&] (u64 lidx) {
            f32* lhs_slice = lhs_section + lidx * total_nb_panels_wing;
            const f32* vwing_slice = vwing_section + lidx + lidx / m_col.ns;
            ispc::kernel_influence(m, lhs_slice, colloc, total_nb_panels_wing, vwing_slice, total_nb_vertices_wing, m_col.ns+1, normals, total_nb_panels_wing);
        });

        auto last_row = graph.for_each_index(end_wing, m_col.nb_panels_wing(), [&] (u64 lidx) {
            f32* lhs_slice = lhs_section + lidx * total_nb_panels_wing;
            const f32* vwing_slice = vwing_section + lidx + lidx / m_col.ns;
            ispc::kernel_influence(m, lhs_slice, colloc, total_nb_panels_wing, vwing_slice, total_nb_vertices_wing, m_col.ns+1, normals, total_nb_panels_wing);
        });

        auto cond = graph.emplace([&]{
            return w < m_col.nwa ? 0 : 1; // 0 means continue, 1 means break
        });
        auto wake_pass = graph.for_each_index(zero, m_col.ns, [&] (u64 j) {
            f32* lhs_slice = lhs_section + (j+end_wing) * total_nb_panels_wing;
            const f32* vwake_slice = vwake_section + (m_col.nw - w - 1) * (m_col.ns+1) + j;
            ispc::kernel_influence(m, lhs_slice, colloc, total_nb_panels_wing, vwake_slice, total_nb_vertices_wing, m_col.ns+1, normals, total_nb_panels_wing);
        });
        auto back = graph.emplace([&]{
            w++;
            return 0; // 0 means continue
        });

        begin.precede(wing_pass, last_row);
        wing_pass.precede(end);
        last_row.precede(cond);
        cond.precede(wake_pass, end); // 0 and 1
        wake_pass.precede(back);
        back.precede(cond);
    }

    Executor::get().run(graph).wait();
}

void BackendCPU::rhs_assemble_velocities(f32* rhs, const f32* normals, const f32* velocities) {
    const tiny::ScopedTimer timer("RHS");

    const u64 total_nb_panels_wing = prm.back().off_wing_p + prm.back().nb_panels_wing(); // todo replace raw ptr to views ? 

    // todo: parallelize
    for (u64 i = 0; i < total_nb_panels_wing; i++) {
        rhs[i] = - (
            velocities[i + 0 * total_nb_panels_wing] * normals[i + 0 * total_nb_panels_wing] +
            velocities[i + 1 * total_nb_panels_wing] * normals[i + 1 * total_nb_panels_wing] +
            velocities[i + 2 * total_nb_panels_wing] * normals[i + 2 * total_nb_panels_wing]);
    }
}

void BackendCPU::rhs_assemble_wake_influence() {
    // const tiny::ScopedTimer timer("Wake Influence");
    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto wake_influence = taskflow.for_each_index((u64)0, ns * nc, [&] (u64 lidx) {
        ispc::kernel_wake_influence(&mesh, lidx, dd_data->gamma, dd_data->rhs, sigma_vatistas);
    });
    auto sync = taskflow.placeholder();

    init.precede(wake_influence);
    wake_influence.precede(sync);

    Executor::get().run(taskflow).wait();
}

void BackendCPU::displace_wake_rollup(float dt) {
    const tiny::ScopedTimer timer("Wake Rollup");

    const u64 nc = dd_mesh->nc;
    const u64 ns = dd_mesh->ns;
    const u64 nw = dd_mesh->nw;
    const u64 nwa = dd_mesh->nwa;

    const u64 wake_vertices_begin = (nc + nw - nwa + 1) * (ns+1);
    const u64 wake_vertices_end = (nc + nw + 1) * (ns + 1);

    const f32* rx = dd_data->rollup_vertices + (nc+nw+1)*(ns+1)*0;
    const f32* ry = dd_data->rollup_vertices + (nc+nw+1)*(ns+1)*1;
    const f32* rz = dd_data->rollup_vertices + (nc+nw+1)*(ns+1)*2;

    ispc::Mesh mesh = mesh_to_ispc(dd_mesh);

    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto rollup = taskflow.for_each_index(wake_vertices_begin, wake_vertices_end, [&] (u64 vidx) {
        ispc::kernel_rollup(&mesh, dt, dd_data->rollup_vertices, vidx, dd_data->gamma, sigma_vatistas);
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

void BackendCPU::gamma_shed() {
    // const tiny::ScopedTimer timer("Shed Gamma");
    const u64 nc = dd_mesh->nc;
    const u64 ns = dd_mesh->ns;
    const u64 nw = dd_mesh->nw;
    const u64 nwa = dd_mesh->nwa;
    const u64 nb_panels_wing = nc * ns;
    const u64 wake_row_start = (nc + nw - nwa - 1) * ns;

    std::copy(dd_data->gamma, dd_data->gamma + nb_panels_wing, dd_data->gamma_prev); // store current timestep for delta_gamma
    std::copy(dd_data->gamma + ns * (nc-1), dd_data->gamma + nb_panels_wing, dd_data->gamma + wake_row_start);
}

void BackendCPU::lu_factor(f32* lhs) {
    // const tiny::ScopedTimer timer("Factor");
    const u64 total_nb_panels_wing = prm.back().off_wing_p + prm.back().nb_panels_wing();
    const int32_t n = static_cast<int32_t>(total_nb_panels_wing);
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, lhs, n, d_solver_ipiv);
}

void BackendCPU::lu_solve(f32* lhs, f32* rhs, f32* gamma) {
    // const tiny::ScopedTimer timer("Solve");
    const u64 total_nb_panels_wing = prm.back().off_wing_p + prm.back().nb_panels_wing();
    const int32_t n = static_cast<int32_t>(total_nb_panels_wing);

    memory->copy(MemoryTransfer::DeviceToDevice, gamma, lhs, total_nb_panels_wing * sizeof(f32));

    LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, 1, lhs, n, d_solver_ipiv, gamma, n);
}

f32 BackendCPU::coeff_steady_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) {
    // const tiny::ScopedTimer timer("Compute CL");

    const u64 nc = dd_mesh->nc;
    const u64 ns = dd_mesh->ns;
    const u64 nw = dd_mesh->nw;
    const u64 nwa = dd_mesh->nwa;
    const u64 nb_panels_wing = nc * ns;

    assert(n > 0);
    assert(j >= 0 && j+n <= dd_mesh->ns);
    
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

f32 BackendCPU::coeff_unsteady_cl(const linalg::alias::float3& freestream, const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) {
    assert(n > 0);
    assert(j >= 0 && j+n <= dd_mesh->ns);

    const u64 nc = dd_mesh->nc;
    const u64 ns = dd_mesh->ns;
    const u64 nw = dd_mesh->nw;
    const u64 nwa = dd_mesh->nwa;
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

linalg::alias::float3 BackendCPU::coeff_steady_cm(
    const FlowData& flow,
    const f32 area,
    const f32 chord,
    const u64 j,
    const u64 n)
{
    assert(n > 0);
    assert(j >= 0 && j+n <= dd_mesh->ns);

    const u64 nc = dd_mesh->nc;
    const u64 ns = dd_mesh->ns;
    const u64 nw = dd_mesh->nw;
    const u64 nwa = dd_mesh->nwa;
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

f32 BackendCPU::coeff_steady_cd(
    const FlowData& flow,
    const f32 area,
    const u64 j,
    const u64 n) 
{
    tiny::ScopedTimer timer("Compute CD");
    assert(n > 0);
    assert(j >= 0 && j+n <= dd_mesh->ns);
    // std::fill(dd_data->trefftz_buffer, dd_data->trefftz_buffer + hd_mesh->ns, 0.0f);
    ispc::Mesh mesh = mesh_to_ispc(dd_mesh);
    f32 cd = ispc::kernel_trefftz_cd2(&mesh, dd_data->gamma, j, n, sigma_vatistas);
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
    std::fill(dd_data->local_velocities + 0 * nb_panels_wing, dd_data->local_velocities + 1 * nb_panels_wing, vel.x);
    std::fill(dd_data->local_velocities + 1 * nb_panels_wing, dd_data->local_velocities + 2 * nb_panels_wing, vel.y);
    std::fill(dd_data->local_velocities + 2 * nb_panels_wing, dd_data->local_velocities + 3 * nb_panels_wing, vel.z);
}

void BackendCPU::set_velocities(const f32* vels) {
    const u64 nb_panels_wing = mesh_nb_panels_wing(hd_mesh);
    memory->hd_copy(dd_data->local_velocities, vels, 3 * nb_panels_wing * sizeof(f32));
}

// TODO: change this to use the per panel local alpha (in global frame)
void BackendCPU::mesh_metrics(const f32 alpha_rad, const f32* verts_wing, f32* colloc, f32* normals, f32* areas) {
    // parallel for
    for (const auto& p : prm) {

        const f32* vx = PTR_V(verts_wing + p.off_wing_v, p.nc, p.ns, 0,0,0);
        const f32* vy = PTR_V(verts_wing + p.off_wing_v, p.nc, p.ns, 0,0,1);
        const f32* vz = PTR_V(verts_wing + p.off_wing_v, p.nc, p.ns, 0,0,2);
        f32* nx = PTR_P(normals + p.off_wing_p, p.nc, p.ns, 0,0,0);
        f32* ny = PTR_P(normals + p.off_wing_p, p.nc, p.ns, 0,0,1);
        f32* nz = PTR_P(normals + p.off_wing_p, p.nc, p.ns, 0,0,2);
        f32* cx = PTR_P(colloc + p.off_wing_p, p.nc, p.ns, 0,0,0);
        f32* cy = PTR_P(colloc + p.off_wing_p, p.nc, p.ns, 0,0,1);
        f32* cz = PTR_P(colloc + p.off_wing_p, p.nc, p.ns, 0,0,2);

        // parallel for
        for (u64 i = 0; i < p.nc; i++) {
            // inner vectorized loop
            for (u64 j = 0; j < p.ns; j++) {
                const u64 lidx = i * (p.ns) + j;
                const u64 v0 = (i+0) * (p.ns+1) + j;
                const u64 v1 = (i+0) * (p.ns+1) + j + 1;
                const u64 v2 = (i+1) * (p.ns+1) + j + 1;
                const u64 v3 = (i+1) * (p.ns+1) + j;

                const linalg::alias::float3 vertex0{vx[v0], vy[v0], vz[v0]};
                const linalg::alias::float3 vertex1{vx[v1], vy[v1], vz[v1]};
                const linalg::alias::float3 vertex2{vx[v2], vy[v2], vz[v2]};
                const linalg::alias::float3 vertex3{vx[v3], vy[v3], vz[v3]};

                const linalg::alias::float3 normal_vec = linalg::normalize(linalg::cross(vertex3 - vertex1, vertex2 - vertex0));
                nx[lidx] = normal_vec.x;
                ny[lidx] = normal_vec.y;
                nz[lidx] = normal_vec.z;

                // 3 vectors f (P0P3), b (P0P2), e (P0P1) to compute the area:
                // area = 0.5 * (||f x b|| + ||b x e||)
                // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
                const linalg::alias::float3 vec_f = vertex3 - vertex0;
                const linalg::alias::float3 vec_b = vertex2 - vertex0;
                const linalg::alias::float3 vec_e = vertex1 - vertex0;

                (areas + p.off_wing_p)[lidx] = 0.5f * (linalg::length(linalg::cross(vec_f, vec_b)) + linalg::length(linalg::cross(vec_b, vec_e)));
                
                // High AoA correction (Aerodynamic Optimization of Aircraft Wings Using a Coupled VLM2.5D RANS Approach) Eq 3.4 p21
                // https://publications.polymtl.ca/2555/1/2017_MatthieuParenteau.pdf
                const f32 factor = (alpha_rad < EPS_f) ? 0.5f : 0.5f * (alpha_rad / (std::sin(alpha_rad) + EPS_f));
                const linalg::alias::float3 chord_vec = 0.5f * (vertex2 + vertex3 - vertex0 - vertex1);
                const linalg::alias::float3 colloc_pt = 0.5f * (vertex0 + vertex1) + factor * chord_vec;
                
                cx[lidx] = colloc_pt.x;
                cy[lidx] = colloc_pt.y;
                cz[lidx] = colloc_pt.z;
            }
        }
    }
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
    assert(i + m <= dd_mesh->nc);
    assert(j + n <= dd_mesh->ns);
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

// TODO: change linalg to use std::array as underlying storage
void linalg_to_flat(const linalg::alias::float4x4& m, float* flat_m) {
    flat_m[0] = m.x.x;
    flat_m[1] = m.x.y;
    flat_m[2] = m.x.z;
    flat_m[3] = m.x.w;
    flat_m[4] = m.y.x;
    flat_m[5] = m.y.y;
    flat_m[6] = m.y.z;
    flat_m[7] = m.y.w;
    flat_m[8] = m.z.x;
    flat_m[9] = m.z.y;
    flat_m[10] = m.z.z;
    flat_m[11] = m.z.w;
    flat_m[12] = m.w.x;
    flat_m[13] = m.w.y;
    flat_m[14] = m.w.z;
    flat_m[15] = m.w.w;
}

void BackendCPU::displace_wing_and_shed(const std::vector<linalg::alias::float4x4>& transforms, f32* verts_wing, f32* verts_wing_init, f32* verts_wake) {
    // const tiny::ScopedTimer t("Mesh::move");
    assert(transforms.size() == prm.size());

    f32 transform[16]; // col major 4x4 matrix

    // TODO: parallel for
    for (u64 i = 0; i < prm.size(); i++) {
        const auto& p = prm[i];
        assert(prm.nwa < prm.nw);
        f32* vwing = verts_wing + p.off_wing_v;
        f32* vwake = verts_wake + p.off_wake_v;
    
        // Shed wake before moving
        memory->copy(MemoryTransfer::DeviceToDevice, PTR_V(vwake, p.nw, p.ns, p.nw-p.nwa,0,0), PTR_V(vwing, p.nc, p.ns, p.nc, 0, 0), (p.ns+1)*sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, PTR_V(vwake, p.nw, p.ns, p.nw-p.nwa,0,1), PTR_V(vwing, p.nc, p.ns, p.nc, 0, 1), (p.ns+1)*sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, PTR_V(vwake, p.nw, p.ns, p.nw-p.nwa,0,2), PTR_V(vwing, p.nc, p.ns, p.nc, 0, 2), (p.ns+1)*sizeof(f32));
        
        linalg_to_flat(transforms[i], &transform[0]);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 4, prm.nb_vertices_wing(), 4, 1.0f, &transform[0], 4, verts_wing_init, prm.nb_vertices_wing(), 0.0f, verts_wing, prm.nb_vertices_wing());

        // Copy new trailing edge vertices on the wake buffer // CAREFUL OVERLAP
        memory->copy(MemoryTransfer::DeviceToDevice, PTR_V(vwake, p.nw, p.ns, p.nw-p.nwa-1,0,0), PTR_V(vwing, p.nc, p.ns, p.nc, 0, 0), (p.ns+1)*sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, PTR_V(vwake, p.nw, p.ns, p.nw-p.nwa-1,0,1), PTR_V(vwing, p.nc, p.ns, p.nc, 0, 1), (p.ns+1)*sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, PTR_V(vwake, p.nw, p.ns, p.nw-p.nwa-1,0,2), PTR_V(vwing, p.nc, p.ns, p.nc, 0, 2), (p.ns+1)*sizeof(f32));
        
        p.nwa++;
    }
}

// Temporary
void BackendCPU::update_wake(const linalg::alias::float3& freestream) {
    const f32 chord_root = 1.0f;
    const f32 off_x = freestream.x * 100.0f * chord_root;
    const f32 off_y = freestream.y * 100.0f * chord_root;
    const f32 off_z = freestream.z * 100.0f * chord_root;

    f32* vx = PTR_MESH_V(dd_mesh, dd_mesh->nc,0,0);
    f32* vy = PTR_MESH_V(dd_mesh, dd_mesh->nc,0,1);
    f32* vz = PTR_MESH_V(dd_mesh, dd_mesh->nc,0,2);

    for (u64 i = 0; i < dd_mesh->ns+1; ++i) {
        vx[i + dd_mesh->ns+1] = vx[i] + off_x;
        vy[i + dd_mesh->ns+1] = vy[i] + off_y;
        vz[i + dd_mesh->ns+1] = vz[i] + off_z;
    }
    dd_mesh->nwa = 1;
}

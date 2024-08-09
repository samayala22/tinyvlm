#include "vlm_backend_cpu.hpp"
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

/// @brief Memory manager implementation for the CPU backend
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

/// @brief Compute the gamma_delta vector
/// @details
/// Compute the gamma_delta vector of the VLM system (\Delta\Gamma = \Gamma_{i,j} - \Gamma_{i-1,j})
/// The vector is computed for each lifting surface of the system
/// @param gamma_delta gamma_delta vector
/// @param gamma gamma vector
void BackendCPU::gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) {
    assert(gamma_delta.layout.dims() == 1);
    assert(gamma.layout.dims() == 1);
    
    tf::Taskflow graph;

    auto begin = graph.placeholder();
    auto end = graph.placeholder();

    for (const auto& surf : gamma_delta.layout.surfaces())  {
        f32* s_gamma_delta = gamma_delta.ptr + surf.offset;
        const f32* s_gamma = gamma.ptr + surf.offset;
        tf::Task first_row = graph.emplace([=]{
            memory->copy(MemoryTransfer::DeviceToDevice, s_gamma_delta, s_gamma, surf.ns * sizeof(*s_gamma_delta));
        });
        tf::Task remaining_rows = graph.for_each_index((u64)1,surf.nc, [=] (u64 b, u64 e) {
            for (u64 i = b; i < e; i++) {
                for (u64 j = 0; j < surf.ns; j++) {
                    s_gamma_delta[i*surf.ns + j] = s_gamma[i*surf.ns + j] - s_gamma[(i-1)*surf.ns + j];
                }
            }
        });

        begin.precede(first_row);
        begin.precede(remaining_rows);
        first_row.precede(end);
        remaining_rows.precede(end);
    };

    Executor::get().run(graph).wait();
    // graph.dump(std::cout);
}

// void kernel_inf(u64 m, f32* lhs, f32* influenced, u64 influenced_ld, f32* influencer, u64 influencer_ld, u64 influencer_rld, f32* normals, u64 normals_ld);

/// @brief Assemble the left hand side matrix
/// @details
/// Assemble the left hand side matrix of the VLM system. The matrix is
/// assembled in column major order. The matrix is assembled for each lifting
/// surface of the system
/// @param lhs left hand side matrix
/// @param colloc collocation points for all surfaces
/// @param normals normals of all surfaces
/// @param verts_wing vertices of the wing surfaces
/// @param verts_wake vertices of the wake surfaces
/// @param iteration iteration number (VLM = 1, UVLM = [0 ... N tsteps])
void BackendCPU::lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<u32>& condition, u32 iteration) {
    // tiny::ScopedTimer timer("LHS");

    assert(condition.size() == colloc.layout.surfaces().size());
    std::fill(condition.begin(), condition.end(), 0); // reset conditon increment vars

    tf::Taskflow graph;

    auto begin = graph.placeholder();
    auto end = graph.placeholder();

    for (u32 i = 0; i < colloc.layout.surfaces().size(); i++) {
        f32* lhs_section = lhs.ptr + colloc.layout.offset(i) * lhs.layout.stride();
        
        f32* vwing_section = verts_wing.ptr + verts_wing.layout.offset(i);
        f32* vwake_section = verts_wake.ptr + verts_wake.layout.offset(i);
        const u64 zero = 0;
        const u64 end_wing = (colloc.layout.nc(i) - 1) * colloc.layout.ns(i);
        
        auto wing_pass = graph.for_each_index(zero, end_wing, [&, lhs_section, vwing_section, i] (u64 lidx) {
            f32* lhs_slice = lhs_section + lidx * lhs.layout.stride();
            f32* vwing_slice = vwing_section + lidx + lidx / colloc.layout.ns(i);
            ispc::kernel_influence(lhs.layout.m(), lhs_slice, colloc.ptr, colloc.layout.stride(), vwing_slice, verts_wing.layout.stride(), verts_wing.layout.ns(i), normals.ptr, normals.layout.stride(), sigma_vatistas);
        });

        auto last_row = graph.for_each_index(end_wing, colloc.layout.ns(i) * colloc.layout.nc(i), [&, lhs_section, vwing_section, i] (u64 lidx) {
            f32* lhs_slice = lhs_section + lidx * lhs.layout.stride();
            f32* vwing_slice = vwing_section + lidx + lidx / colloc.layout.ns(i);
            ispc::kernel_influence(lhs.layout.m(), lhs_slice, colloc.ptr, colloc.layout.stride(), vwing_slice, verts_wing.layout.stride(), verts_wing.layout.ns(i), normals.ptr, normals.layout.stride(), sigma_vatistas);
        });

        auto cond = graph.emplace([&, iteration, i]{
            return condition[i] < iteration ? 0 : 1; // 0 means continue, 1 means break (exit loop)
        });
        auto wake_pass = graph.for_each_index(zero, colloc.layout.ns(i), [&, lhs_section, vwake_section, end_wing, i] (u64 j) {
            f32* lhs_slice = lhs_section + (j+end_wing) * lhs.layout.stride();
            f32* vwake_slice = vwake_section + (verts_wake.layout.nc(i) - condition[i] - 2) * verts_wake.layout.ns(i) + j;
            ispc::kernel_influence(lhs.layout.m(), lhs_slice, colloc.ptr, colloc.layout.stride(), vwake_slice, verts_wake.layout.stride(), verts_wake.layout.ns(i), normals.ptr, normals.layout.stride(), sigma_vatistas);
        });
        auto back = graph.emplace([&, i]{
            condition[i]++;
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

/// @brief Add velocity contributions to the right hand side vector
/// @details
/// Add velocity contributions to the right hand side vector of the VLM system
/// @param rhs right hand side vector
/// @param normals normals of all surfaces
/// @param velocities displacement velocities of all surfaces
void BackendCPU::rhs_assemble_velocities(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& velocities) {
    // const tiny::ScopedTimer timer("RHS");
    assert(rhs.layout.stride() == rhs.size()); // single dim
    assert(rhs.layout.stride() == normals.layout.stride());
    assert(rhs.layout.stride() == velocities.layout.stride());
    assert(rhs.layout.dims() == 1);

    // todo: parallelize
    for (u64 i = 0; i < rhs.size(); i++) {
        rhs[i] += - (
            velocities[i + 0 * velocities.layout.stride()] * normals[i + 0 * normals.layout.stride()] +
            velocities[i + 1 * velocities.layout.stride()] * normals[i + 1 * normals.layout.stride()] +
            velocities[i + 2 * velocities.layout.stride()] * normals[i + 2 * normals.layout.stride()]);
    }
}

void BackendCPU::rhs_assemble_wake_influence(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& gamma, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wake, u32 iteration) {
    // const tiny::ScopedTimer timer("Wake Influence");
    assert(rhs.layout.stride() == rhs.size()); // single dim

    tf::Taskflow taskflow;

    auto begin = taskflow.placeholder();
    auto end = taskflow.placeholder();

    auto wake_influence = taskflow.for_each_index((u64)0, rhs.layout.stride(), [&] (u64 idx) {
        for (u32 i = 0; i < rhs.layout.surfaces().size(); i++) {
            ispc::kernel_wake_influence(colloc.ptr + idx, colloc.layout.stride(), normals.ptr + idx, normals.layout.stride(), verts_wake.ptr + verts_wake.layout.offset(i), verts_wake.layout.stride(), verts_wake.layout.nc(i), verts_wake.layout.ns(i), gamma.ptr + idx, rhs.ptr + idx, sigma_vatistas, iteration);
        }
    }).name("RHS Wake Influence");

    begin.precede(wake_influence);
    wake_influence.precede(end);

    Executor::get().run(taskflow).wait();
}

void BackendCPU::displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, u32 iteration) {
    // const tiny::ScopedTimer timer("Wake Rollup");
    tf::Taskflow taskflow;

    auto begin = taskflow.placeholder();
    auto end = taskflow.placeholder();

    for (u64 m = 0; m < verts_wake.layout.surfaces().size(); m++) {
        const u64 wake_begin = (verts_wake.layout.nc(m) - iteration) * (verts_wake.layout.ns(m));
        const u64 wake_end = verts_wake.layout.nc(m) * (verts_wake.layout.ns(m));
        auto rollup = taskflow.for_each_index(wake_begin, wake_end, [&] (u64 vidx) {
            for (u64 i = 0; i < verts_wake.layout.surfaces().size(); i++) {
                ispc::kernel_rollup(verts_wake.ptr + verts_wake.layout.offset(i), verts_wake.layout.stride(), verts_wake.layout.nc(i), verts_wake.layout.ns(i), vidx, verts_wing.ptr + verts_wing.layout.offset(i), verts_wing.layout.stride(), verts_wing.layout.nc(i), verts_wing.layout.ns(i), wake_rollup.ptr + wake_rollup.layout.offset(i), wake_rollup.layout.stride(), gamma_wing.ptr + gamma_wing.layout.offset(i), gamma_wake.ptr + gamma_wake.layout.offset(i), sigma_vatistas, dt, iteration);
            }
        });
        auto copy = taskflow.emplace([&, wake_begin, wake_end, m]{
            memory->copy(MemoryTransfer::DeviceToDevice, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 0*verts_wake.layout.stride(), wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 0*wake_rollup.layout.stride(), (wake_end - wake_begin) * sizeof(f32));
            memory->copy(MemoryTransfer::DeviceToDevice, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 1*verts_wake.layout.stride(), wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 1*wake_rollup.layout.stride(), (wake_end - wake_begin) * sizeof(f32));
            memory->copy(MemoryTransfer::DeviceToDevice, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 2*verts_wake.layout.stride(), wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 2*wake_rollup.layout.stride(), (wake_end - wake_begin) * sizeof(f32));
        });
        begin.precede(rollup);
        rollup.precede(copy);
        copy.precede(end);
    }

    Executor::get().run(taskflow).wait();
}

void BackendCPU::gamma_shed(View<f32, MultiSurface>& gamma_wing, View<f32, MultiSurface>& gamma_wing_prev, View<f32, MultiSurface>& gamma_wake, u32 iteration) {
    // const tiny::ScopedTimer timer("Shed Gamma");

    memory->copy(MemoryTransfer::DeviceToDevice, gamma_wing_prev.ptr, gamma_wing.ptr, gamma_wing.size_bytes());
    for (u64 i = 0; i < gamma_wake.layout.surfaces().size(); i++) {
        assert(iteration < gamma_wake.layout.nc(i));
        f32* gamma_wake_ptr = gamma_wake.ptr + gamma_wake.layout.offset(i) + (gamma_wake.layout.nc(i) - iteration - 1) * gamma_wake.layout.ns(i);
        f32* gamma_wing_ptr = gamma_wing.ptr + gamma_wing.layout.offset(i) + (gamma_wing.layout.nc(i) - 1) * gamma_wing.layout.ns(i); // last row
        memory->copy(MemoryTransfer::DeviceToDevice, gamma_wake_ptr, gamma_wing_ptr, gamma_wing.layout.ns(i) * sizeof(f32));
    }
}

// TODO: consider moving this buffer to be simulation side rather than backend side
void BackendCPU::lu_allocate(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) {
    d_solver_ipiv = (i32*)memory->alloc(MemoryLocation::Device, sizeof(i32) * lhs.layout.m());
}

void BackendCPU::lu_factor(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) {
    // const tiny::ScopedTimer timer("Factor");
    assert(lhs.layout.m() == lhs.layout.n()); // square matrix
    const int32_t n = static_cast<int32_t>(lhs.layout.n());
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, lhs.ptr, n, d_solver_ipiv);
}

void BackendCPU::lu_solve(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, View<f32, MultiSurface>& rhs, View<f32, MultiSurface>& gamma) {
    // const tiny::ScopedTimer timer("Solve");
    const int32_t n = static_cast<int32_t>(lhs.layout.n());

    memory->copy(MemoryTransfer::DeviceToDevice, gamma.ptr, rhs.ptr, rhs.size_bytes());

    LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, 1, lhs.ptr, n, d_solver_ipiv, gamma.ptr, n);
}

f32 BackendCPU::coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow, f32 area) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cl = 0.0f;

    const u64 nc = gamma_delta.layout.surface().nc;
    const u64 ns = gamma_delta.layout.surface().ns;
    for (u64 i = 0; i < nc; i++) {
        for (u64 j = 0; j < ns; j++) {
            const u64 v0 = (i+0) * verts_wing.layout.ld() + j;
            const u64 v1 = (i+0) * verts_wing.layout.ld() + j + 1;
            const linalg::alias::float3 vertex0{verts_wing.ptr[0*verts_wing.layout.stride() + v0], verts_wing.ptr[1*verts_wing.layout.stride() + v0], verts_wing.ptr[2*verts_wing.layout.stride() + v0]}; // upper left
            const linalg::alias::float3 vertex1{verts_wing.ptr[0*verts_wing.layout.stride() + v1], verts_wing.ptr[1*verts_wing.layout.stride() + v1], verts_wing.ptr[2*verts_wing.layout.stride() + v1]}; // upper right
            // const linalg::alias::float3 v3 = mesh.get_v3(li);
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = vertex1 - vertex0;
            // const linalg::alias::float3 local_left_chord = linalg::normalize(v3 - v0);
            // const linalg::alias::float3 projected_vector = linalg::dot(dl, local_left_chord) * local_left_chord;
            // dl -= projected_vector; // orthogonal leading edge vector
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * gamma_delta.ptr[i * gamma_delta.layout.ld() + j];
            cl += linalg::dot(force, flow.lift_axis); // projection on the body lift axis
        }
    }
    cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;

    return cl;
}

f32 BackendCPU::coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (u64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i, 0, verts_wing.layout.nc(i), 0, verts_wing.layout.ns(i));
        const auto gamma_delta_local = gamma_delta.layout.subview(gamma_delta.ptr, i, 0, gamma_delta.layout.nc(i), 0, gamma_delta.layout.ns(i));
        f32 area_local = 0.f;
        const f32* areas_local = areas.ptr + areas.layout.offset(i);
        for (u64 j = 0; j < areas.layout.surface(i).size(); j++) {
            area_local += areas_local[j];
        }

        const f32 wing_cl = coeff_steady_cl_single(verts_wing_local, gamma_delta_local, flow, area_local);
        cl += wing_cl * area_local;
        total_area += area_local;
    }
    cl /= total_area;
    return cl;
}

// f32 BackendCPU::coeff_unsteady_cl(const linalg::alias::float3& freestream, const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) {
//     assert(n > 0);
//     assert(j >= 0 && j+n <= dd_mesh->ns);

//     const u64 nc = dd_mesh->nc;
//     const u64 ns = dd_mesh->ns;
//     const u64 nw = dd_mesh->nw;
//     const u64 nwa = dd_mesh->nwa;
//     const u64 nb_panels_wing = nc * ns;
    
//     f32 cl = 0.0f;
//     const f32 rho = 1.0f; // TODO: remove hardcoded rho
//     const linalg::alias::float3 span_axis{dd_mesh->frame + 4};
//     const linalg::alias::float3 lift_axis = linalg::normalize(linalg::cross(freestream, span_axis));

//     for (u64 u = 0; u < nc; u++) {
//         for (u64 v = j; v < j + n; v++) {
//             const u64 li = u * ns + v; // linear index

//             linalg::alias::float3 V{vel.x[li], vel.y[li], vel.z[li]}; // local velocity (freestream + displacement vel)

//             const linalg::alias::float3 v0{*PTR_MESH_V(dd_mesh, u, v, 0), *PTR_MESH_V(dd_mesh, u, v, 1), *PTR_MESH_V(dd_mesh, u, v, 2)}; // upper left
//             const linalg::alias::float3 v1{*PTR_MESH_V(dd_mesh, u, v+1, 0), *PTR_MESH_V(dd_mesh, u, v+1, 1), *PTR_MESH_V(dd_mesh, u, v+1, 2)}; // upper right
//             const linalg::alias::float3 normal{*PTR_MESH_N(dd_mesh, u, v, 0), *PTR_MESH_N(dd_mesh, u, v, 1), *PTR_MESH_N(dd_mesh, u, v, 2)}; // normal

//             linalg::alias::float3 force = {0.0f, 0.0f, 0.0f};
//             const f32 gamma_dt = (dd_data->gamma[li] - dd_data->gamma_prev[li]) / dt; // backward difference

//             // Joukowski method
//             force += rho * dd_data->delta_gamma[li] * linalg::cross(V, v1 - v0);
//             force += rho * gamma_dt * dd_mesh->area[li] * normal;

//             // Katz Plotkin method
//             // linalg::alias::float3 delta_p = {0.0f, 0.0f, 0.0f};
//             // const f32 delta_gamma_i = (u == 0) ? gamma[li] : gamma[li] - gamma[(u-1) * mesh.ns + v];
//             // const f32 delta_gamma_j = (v == 0) ? gamma[li] : gamma[li] - gamma[u * mesh.ns + v - 1];
//             // delta_p += rho * linalg::dot(freestream, linalg::normalize(v1 - v0)) * delta_gamma_j / mesh.panel_width_y(u, v);
//             // delta_p += rho * linalg::dot(freestream, linalg::normalize(v3 - v0)) * delta_gamma_i / mesh.panel_length(u, v);
//             // delta_p += gamma_dt;
//             // force = (delta_p * mesh.area[li]) * normal;

//             // force /= linalg::length2(freestream);
            
//             cl += linalg::dot(force, lift_axis);
//         }
//     }
//     cl /= 0.5f * rho * linalg::length2(freestream) * area;

//     return cl;
// }

// linalg::alias::float3 BackendCPU::coeff_steady_cm(
//     const FlowData& flow,
//     const f32 area,
//     const f32 chord,
//     const u64 j,
//     const u64 n)
// {
//     assert(n > 0);
//     assert(j >= 0 && j+n <= dd_mesh->ns);

//     const u64 nc = dd_mesh->nc;
//     const u64 ns = dd_mesh->ns;
//     const u64 nw = dd_mesh->nw;
//     const u64 nwa = dd_mesh->nwa;
//     const u64 nb_panels_wing = nc * ns;

//     linalg::alias::float3 cm(0.f, 0.f, 0.f);
//     const linalg::alias::float3 ref_pt{dd_mesh->frame + 12}; // frame origin as moment pt

//     for (u64 u = 0; u < nc; u++) {
//         for (u64 v = j; v < j + n; v++) {
//             const u64 li = u * ns + v; // linear index
//             const linalg::alias::float3 v0{*PTR_MESH_V(dd_mesh, u, v, 0), *PTR_MESH_V(dd_mesh, u, v, 1), *PTR_MESH_V(dd_mesh, u, v, 2)}; // upper left
//             const linalg::alias::float3 v1{*PTR_MESH_V(dd_mesh, u, v+1, 0), *PTR_MESH_V(dd_mesh, u, v+1, 1), *PTR_MESH_V(dd_mesh, u, v+1, 2)}; // upper right
//             // Leading edge vector pointing outward from wing root
//             const linalg::alias::float3 dl = v1 - v0;
//             // Distance from the center of leading edge to the reference point
//             const linalg::alias::float3 dst_to_ref = ref_pt - 0.5f * (v0 + v1);
//             // Distance from the center of leading edge to the reference point
//             const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * dd_data->delta_gamma[li];
//             cm += linalg::cross(force, dst_to_ref);
//         }
//     }
//     cm /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area * chord;
//     return cm;
// }

f32 BackendCPU::coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const View<f32, SingleSurface>& gamma_wake, const FlowData& flow, f32 area) {
    // tiny::ScopedTimer timer("Compute CD");
    f32 cd = ispc::kernel_trefftz_cd(verts_wake.ptr, verts_wake.layout.stride(), verts_wake.layout.surface().nc, verts_wake.layout.surface().ns, gamma_wake.ptr, sigma_vatistas);
    cd /= linalg::length2(flow.freestream) * area;
    return cd;
}

f32 BackendCPU::coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& gamma_wake, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cd = 0.0f;
    f32 total_area = 0.0f;
    for (u64 i = 0; i < verts_wake.layout.surfaces().size(); i++) {
        const auto verts_wake_local = verts_wake.layout.subview(verts_wake.ptr, i, 0, verts_wake.layout.nc(i), 0, verts_wake.layout.ns(i));
        const auto gamma_wake_local = gamma_wake.layout.subview(gamma_wake.ptr, i, 0, gamma_wake.layout.nc(i), 0, gamma_wake.layout.ns(i));
        f32 area_local = 0.f;
        const f32* areas_local = areas.ptr + areas.layout.offset(i);
        for (u64 j = 0; j < areas.layout.surface(i).size(); j++) {
            area_local += areas_local[j];
        }

        const f32 wing_cd = coeff_steady_cd_single(verts_wake_local, gamma_wake_local, flow, area_local);
        cd += wing_cd * area_local;
        total_area += area_local;
    }
    cd /= total_area;
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

// TODO: change this to use the per panel local alpha (in global frame)
void BackendCPU::mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& colloc, View<f32, MultiSurface>& normals, View<f32, MultiSurface>& areas) {
    // parallel for
    for (int m = 0; m < colloc.layout.surfaces().size(); m++) {
        const f32* verts_wing_ptr = verts_wing.ptr + verts_wing.layout.offset(m);
        f32* colloc_ptr = colloc.ptr + colloc.layout.offset(m);
        f32* normals_ptr = normals.ptr + normals.layout.offset(m);
        f32* areas_ptr = areas.ptr + areas.layout.offset(m);
        // parallel for
        for (u64 i = 0; i < colloc.layout.nc(m); i++) {
            // inner vectorized loop
            for (u64 j = 0; j < colloc.layout.ns(m); j++) {
                const u64 lidx = i * colloc.layout.ns(m) + j;
                const u64 v0 = (i+0) * verts_wing.layout.ns(m) + j;
                const u64 v1 = (i+0) * verts_wing.layout.ns(m) + j + 1;
                const u64 v2 = (i+1) * verts_wing.layout.ns(m) + j + 1;
                const u64 v3 = (i+1) * verts_wing.layout.ns(m) + j;

                const linalg::alias::float3 vertex0{verts_wing_ptr[0*verts_wing.layout.stride() + v0], verts_wing_ptr[1*verts_wing.layout.stride() + v0], verts_wing_ptr[2*verts_wing.layout.stride() + v0]}; // upper left
                const linalg::alias::float3 vertex1{verts_wing_ptr[0*verts_wing.layout.stride() + v1], verts_wing_ptr[1*verts_wing.layout.stride() + v1], verts_wing_ptr[2*verts_wing.layout.stride() + v1]}; // upper right
                const linalg::alias::float3 vertex2{verts_wing_ptr[0*verts_wing.layout.stride() + v2], verts_wing_ptr[1*verts_wing.layout.stride() + v2], verts_wing_ptr[2*verts_wing.layout.stride() + v2]}; // lower right
                const linalg::alias::float3 vertex3{verts_wing_ptr[0*verts_wing.layout.stride() + v3], verts_wing_ptr[1*verts_wing.layout.stride() + v3], verts_wing_ptr[2*verts_wing.layout.stride() + v3]}; // lower left

                const linalg::alias::float3 normal_vec = linalg::normalize(linalg::cross(vertex3 - vertex1, vertex2 - vertex0));
                normals_ptr[0*normals.layout.stride() + lidx] = normal_vec.x;
                normals_ptr[1*normals.layout.stride() + lidx] = normal_vec.y;
                normals_ptr[2*normals.layout.stride() + lidx] = normal_vec.z;

                // 3 vectors f (P0P3), b (P0P2), e (P0P1) to compute the area:
                // area = 0.5 * (||f x b|| + ||b x e||)
                // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
                const linalg::alias::float3 vec_f = vertex3 - vertex0;
                const linalg::alias::float3 vec_b = vertex2 - vertex0;
                const linalg::alias::float3 vec_e = vertex1 - vertex0;

                areas_ptr[lidx] = 0.5f * (linalg::length(linalg::cross(vec_f, vec_b)) + linalg::length(linalg::cross(vec_b, vec_e)));
                
                // High AoA correction (Aerodynamic Optimization of Aircraft Wings Using a Coupled VLM2.5D RANS Approach) Eq 3.4 p21
                // https://publications.polymtl.ca/2555/1/2017_MatthieuParenteau.pdf
                const f32 factor = (alpha_rad < EPS_f) ? 0.5f : 0.5f * (alpha_rad / (std::sin(alpha_rad) + EPS_f));
                const linalg::alias::float3 chord_vec = 0.5f * (vertex2 + vertex3 - vertex0 - vertex1);
                const linalg::alias::float3 colloc_pt = 0.5f * (vertex0 + vertex1) + factor * chord_vec;
                
                colloc_ptr[0*colloc.layout.stride() + lidx] = colloc_pt.x;
                colloc_ptr[1*colloc.layout.stride() + lidx] = colloc_pt.y;
                colloc_ptr[2*colloc.layout.stride() + lidx] = colloc_pt.z;
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
f32 BackendCPU::mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) {
    // Leading edge vertex
    f32* leading_edge_ptr = verts_wing.ptr;
    f32* trailing_edge_ptr = verts_wing.ptr + (verts_wing.layout.surface().nc - 1) * verts_wing.layout.surface().ns;

    f32 mac = 0.0f;
    // loop over panel chordwise sections in spanwise direction
    // Note: can be done optimally with vertical fused simd
    for (u64 v = 0; v < areas.layout.surface().ns; v++) {
        // left and right chord lengths
        const f32 dx0 = trailing_edge_ptr[0*verts_wing.layout.stride() + v + 0] - leading_edge_ptr[0*verts_wing.layout.stride() + v + 0];
        const f32 dy0 = trailing_edge_ptr[1*verts_wing.layout.stride() + v + 0] - leading_edge_ptr[1*verts_wing.layout.stride() + v + 0];
        const f32 dz0 = trailing_edge_ptr[2*verts_wing.layout.stride() + v + 0] - leading_edge_ptr[2*verts_wing.layout.stride() + v + 0];
        const f32 dx1 = trailing_edge_ptr[0*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[0*verts_wing.layout.stride() + v + 1];
        const f32 dy1 = trailing_edge_ptr[1*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[1*verts_wing.layout.stride() + v + 1];
        const f32 dz1 = trailing_edge_ptr[2*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[2*verts_wing.layout.stride() + v + 1];
        const f32 c0 = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
        const f32 c1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
        // Panel width
        const f32 dx3 = leading_edge_ptr[0*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[0*verts_wing.layout.stride() + v + 0];
        const f32 dy3 = leading_edge_ptr[1*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[1*verts_wing.layout.stride() + v + 0];
        const f32 dz3 = leading_edge_ptr[2*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[2*verts_wing.layout.stride() + v + 0];
        const f32 width = std::sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

        mac += 0.5f * (c0 * c0 + c1 * c1) * width;
    }
    // Since we divide by half the total wing area (both sides) we dont need to multiply by 2

    f32 wing_area = 0.0f;
    for (u64 i = 0; i < areas.layout.surface().size(); i++) {
        wing_area += areas[i];
    }
    return mac / wing_area;
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

void BackendCPU::displace_wing(const std::vector<linalg::alias::float4x4>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) {
    // const tiny::ScopedTimer t("Mesh::move");
    assert(transforms.size() == verts_wing.layout.surfaces().size());
    assert(verts_wing.layout.size() == verts_wing_init.layout.size());

    f32 transform[16]; // col major 4x4 matrix

    // TODO: parallel for
    for (u64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        f32* vwing_ptr = verts_wing.ptr + verts_wing.layout.offset(i);
        f32* vwing_init_ptr = verts_wing_init.ptr + verts_wing.layout.offset(i);

        linalg_to_flat(transforms[i], &transform[0]);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 4, static_cast<i32>(verts_wing.layout.surface(i).size()), 4, 1.0f, &transform[0], 4, vwing_init_ptr, static_cast<i32>(verts_wing_init.layout.stride()), 0.0f, vwing_ptr, static_cast<i32>(verts_wing.layout.stride()));
    }
}

void BackendCPU::wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, u32 iteration) {
    assert(verts_wing.layout.surfaces().size() == verts_wake.layout.surfaces().size());

    for (u64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        assert(iteration < verts_wake.layout.nc(i));
        f32* vwing = verts_wing.ptr + verts_wing.layout.offset(i) + (verts_wing.layout.nc(i) - 1) * verts_wing.layout.ns(i);
        f32* vwake = verts_wake.ptr + verts_wake.layout.offset(i) + (verts_wake.layout.nc(i) - iteration - 1) * verts_wake.layout.ns(i);

        memory->copy(MemoryTransfer::DeviceToDevice, vwake + 0*verts_wake.layout.stride(), vwing + 0*verts_wing.layout.stride(), verts_wing.layout.ns(i) * sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, vwake + 1*verts_wake.layout.stride(), vwing + 1*verts_wing.layout.stride(), verts_wing.layout.ns(i) * sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, vwake + 2*verts_wake.layout.stride(), vwing + 2*verts_wing.layout.stride(), verts_wing.layout.ns(i) * sizeof(f32));
    }
}

f32 BackendCPU::mesh_area(const View<f32, SingleSurface>& areas) {
    f32 wing_area = 0.0f;
    for (u64 i = 0; i < areas.layout.surface().size(); i++) {
        wing_area += areas[i];
    }
    return wing_area;
}
#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_memory.hpp"

namespace vlm {

class Backend {
    public:
        const std::unique_ptr<Memory> memory;

        i32* d_solver_info = nullptr;
        i32* d_solver_ipiv = nullptr;
        f32* d_solver_buffer = nullptr;

        f32 sigma_vatistas = 0.0f;
        Backend(std::unique_ptr<Memory> memory) : memory(std::move(memory)) {}
        ~Backend();

        // Kernels that run for all the meshes
        virtual void lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, u32 iteration) = 0;
        virtual void rhs_assemble_velocities(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& velocities) = 0;
        virtual void rhs_assemble_wake_influence(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& gamma) = 0;
        virtual void displace_wake_rollup(float dt, View<f32, MultiSurface>& rollup_vertices, View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma) = 0;
        // virtual void displace_wing(const linalg::alias::float4x4& transform) = 0;
        virtual void displace_wing_and_shed(const std::vector<linalg::alias::float4x4>& transform, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake) = 0;
        virtual void gamma_shed(View<f32, MultiSurface>& gamma, View<f32, MultiSurface>& gamma_prev) = 0;
        virtual void gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) = 0;
        virtual void lu_factor(View<f32, MultiSurface>& lhs) = 0;
        virtual void lu_solve(View<f32, MultiSurface>& lhs, View<f32, MultiSurface>& rhs, View<f32, MultiSurface>& gamma) = 0;
        
        // Per mesh kernels 
        virtual f32 coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow) = 0;
        virtual f32 coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow) = 0;
        virtual f32 coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& local_velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, const linalg::alias::float3& freestream, f32 dt) = 0;
        virtual f32 coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma, const View<f32, MultiSurface>& gamma_prev, const View<f32, MultiSurface>& local_velocities, const View<f32, MultiSurface>& areas, const View<f32, MultiSurface>& normals, const linalg::alias::float3& freestream, f32 dt) = 0;

        // virtual linalg::alias::float3 coeff_steady_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) = 0;
        // virtual f32 coeff_steady_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;

        virtual void mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& colloc, View<f32, MultiSurface>& normals, View<f32, MultiSurface>& areas) = 0;
        // virtual f32 mesh_mac(const u64 j, const u64 n) = 0;
        // virtual f32 mesh_area(const u64 i, const u64 j, const u64 m, const u64 n) = 0;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name);
std::vector<std::string> get_available_backends();

} // namespace vlm

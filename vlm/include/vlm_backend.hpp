#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_allocator.hpp"

namespace vlm {

class Backend {
    public:
        const Allocator allocator;
        std::vector<MeshParams> prm;

        i32* d_solver_info = nullptr;
        i32* d_solver_ipiv = nullptr;
        f32* d_solver_buffer = nullptr;

        f32 sigma_vatistas = 0.0f;
        Backend(const Allocator& allocator) : allocator(allocator) {}
        ~Backend();

        // Kernels that run for all the meshes
        virtual void lhs_assemble(f32* lhs) = 0;
        virtual void rhs_assemble_velocities(f32* rhs, const f32* normals, const f32* velocities) = 0;
        virtual void rhs_assemble_wake_influence(f32* rhs, const f32* gamma) = 0;
        virtual void displace_wake_rollup(float dt, f32* rollup_vertices, f32* verts_wake, const f32* verts_wing, const f32* gamma) = 0;
        // virtual void displace_wing(const linalg::alias::float4x4& transform) = 0;
        virtual void displace_wing_and_shed(const std::vector<linalg::alias::float4x4>& transform, f32* verts_wing, f32* verts_wake) = 0;
        virtual void gamma_shed(f32* gamma, f32* gamma_prev) = 0;
        virtual void gamma_delta(f32* gamma_delta, const f32* gamma) = 0;
        virtual void lu_factor(f32* lhs) = 0;
        virtual void lu_solve(f32* lhs, f32* rhs, f32* gamma) = 0;
        
        // Per mesh kernels
        virtual f32 coeff_steady_cl(const MeshParams& param, const f32* verts_wing, const f32* gamma_delta, const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;
        virtual f32 coeff_unsteady_cl(const MeshParams& param, const f32* verts_wing, const f32* gamma_delta, const f32* gamma, const f32* gamma_prev, const f32* local_velocities, const f32* areas, const f32* normals, const linalg::alias::float3& freestream, f32 dt, const f32 area, const u64 j, const u64 n) = 0;
        virtual linalg::alias::float3 coeff_steady_cm(const MeshParams& param, const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) = 0;
        virtual f32 coeff_steady_cd(const MeshParams& param, const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;

        virtual void mesh_metrics(const f32 alpha) = 0;
        virtual f32 mesh_mac(const MeshParams& param, const u64 j, const u64 n) = 0;
        virtual f32 mesh_area(const MeshParams& param, const u64 i, const u64 j, const u64 m, const u64 n) = 0;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name);
std::vector<std::string> get_available_backends();

} // namespace vlm

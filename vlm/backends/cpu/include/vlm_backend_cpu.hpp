#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendCPU final : public Backend {
    public:
        BackendCPU();
        ~BackendCPU();
        void lhs_assemble(f32* lhs, const f32* colloc, const f32* normals, const f32* verts_wing, const f32* verts_wake) override;
        void rhs_assemble_velocities(f32* rhs, const f32* normals, const f32* velocities) override;
        void rhs_assemble_wake_influence(f32* rhs, const f32* gamma) override;
        void displace_wake_rollup(float dt, f32* rollup_vertices, f32* verts_wake, const f32* verts_wing, const f32* gamma) override;
        // void displace_wing(const linalg::alias::float4x4& transform) override;
        void displace_wing_and_shed(const std::vector<linalg::alias::float4x4>& transform, f32* verts_wing, f32* verts_wake) override;
        void gamma_shed(f32* gamma, f32* gamma_prev) override;
        void gamma_delta(f32* gamma_delta, const f32* gamma) override;
        void lu_factor(f32* lhs) override;
        void lu_solve(f32* lhs, f32* rhs, f32* gamma) override;
        
        // Per mesh kernels
        f32 coeff_steady_cl(const MeshParams& param, const f32* verts_wing, const f32* gamma_delta, const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        f32 coeff_unsteady_cl(const MeshParams& param, const f32* verts_wing, const f32* gamma_delta, const f32* gamma, const f32* gamma_prev, const f32* local_velocities, const f32* areas, const f32* normals, const linalg::alias::float3& freestream, f32 dt, const f32 area, const u64 j, const u64 n) override;
        linalg::alias::float3 coeff_steady_cm(const MeshParams& param, const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;
        f32 coeff_steady_cd(const MeshParams& param, const FlowData& flow, const f32 area, const u64 j, const u64 n) override;

        void mesh_metrics(const f32 alpha_rad, const f32* verts_wing, f32* colloc, f32* normals, f32* areas) override;
        f32 mesh_mac(const MeshParams& param, const u64 j, const u64 n) override;
        f32 mesh_area(const MeshParams& param, const u64 i, const u64 j, const u64 m, const u64 n) override;
};

} // namespace vlm
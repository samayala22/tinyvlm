#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendCPU final : public Backend {
    public:
        BackendCPU();
        ~BackendCPU();
        void lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, u32 iteration) override;
        void rhs_assemble_velocities(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& velocities) override;
        void rhs_assemble_wake_influence(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& gamma, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wake, u32 iteration) override;
        void displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, u32 iteration) override;
        // void displace_wing(const linalg::alias::float4x4& transform) override;
        void displace_wing_and_shed(const std::vector<linalg::alias::float4x4>& transform, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake) override;
        void gamma_shed(View<f32, MultiSurface>& gamma, View<f32, MultiSurface>& gamma_prev) override;
        void gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) override;
        void lu_factor(View<f32, MultiSurface>& lhs) override;
        void lu_solve(View<f32, MultiSurface>& lhs, View<f32, MultiSurface>& rhs, View<f32, MultiSurface>& gamma) override;
        
        // Per mesh kernels
        f32 coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow) override;
        f32 coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow) override;
        f32 coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& local_velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, const linalg::alias::float3& freestream, f32 dt) override;
        f32 coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma, const View<f32, MultiSurface>& gamma_prev, const View<f32, MultiSurface>& local_velocities, const View<f32, MultiSurface>& areas, const View<f32, MultiSurface>& normals, const linalg::alias::float3& freestream, f32 dt) override;
        // linalg::alias::float3 coeff_steady_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;
        // f32 coeff_steady_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;

        void mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& colloc, View<f32, MultiSurface>& normals, View<f32, MultiSurface>& areas) override;
        // f32 mesh_mac(const u64 j, const u64 n) override;
        // f32 mesh_area(const u64 i, const u64 j, const u64 m, const u64 n) override;
};

} // namespace vlm
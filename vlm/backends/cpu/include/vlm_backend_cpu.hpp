#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendCPU final : public Backend {
    public:
        BackendCPU();
        ~BackendCPU();
        void lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<u32>& condition, u32 iteration) override;
        void rhs_assemble_velocities(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& velocities) override;
        void rhs_assemble_wake_influence(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& gamma, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wake, u32 iteration) override;
        void displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, u32 iteration) override;
        void displace_wing(const std::vector<linalg::alias::float4x4>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) override;
        void wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, u32 iteration) override;
        void gamma_shed(View<f32, MultiSurface>& gamma_wing, View<f32, MultiSurface>& gamma_wing_prev, View<f32, MultiSurface>& gamma_wake, u32 iteration) override;
        void gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) override;
        void lu_allocate(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) override;
        void lu_factor(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) override;
        void lu_solve(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, View<f32, MultiSurface>& rhs, View<f32, MultiSurface>& gamma) override;
        
        // Per mesh kernels
        f32 coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow, f32 area) override;
        f32 coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow, const View<f32, MultiSurface>& areas) override;
        f32 coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const View<f32, SingleSurface>& gamma_wake, const FlowData& flow, f32 area) override;
        f32 coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& gamma_wake, const FlowData& flow, const View<f32, MultiSurface>& areas) override;

        // f32 coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& local_velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) override;
        // f32 coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& areas, const View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma, const View<f32, MultiSurface>& gamma_prev, const View<f32, MultiSurface>& local_velocities, const View<f32, MultiSurface>& normals, const linalg::alias::float3& freestream, f32 dt) override;
        // linalg::alias::float3 coeff_steady_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;

        void mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& colloc, View<f32, MultiSurface>& normals, View<f32, MultiSurface>& areas) override;
        f32 mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) override;
};

} // namespace vlm
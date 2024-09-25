#pragma once 

#include "vlm_backend.hpp" // base class
#include "vlm_data.hpp" // TODO: remove ?
#include "vlm_memory.hpp" // View, Layouts

namespace vlm {

class BackendCUDA final : public Backend {
    public:
        BackendCUDA();
        ~BackendCUDA() override;
        void lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<u32>& condition, u32 iteration) override;
        void rhs_assemble_velocities(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& velocities) override;
        void rhs_assemble_wake_influence(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& gamma_wake, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wake, u32 iteration) override;
        void displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, u32 iteration) override;
        void displace_wing(const View<f32, Tensor<3>>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) override;
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
        f32 coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& local_velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) override;
        f32 coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const View<f32, MultiSurface>& normals, const linalg::alias::float3& freestream, f32 dt) override;
        void coeff_unsteady_cl_single_forces(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, View<f32, SingleSurface>& forces, const linalg::alias::float3& freestream, f32 dt) override;
        void coeff_unsteady_cl_multi_forces(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const View<f32, MultiSurface>& normals, View<f32, MultiSurface>& forces, const linalg::alias::float3& freestream, f32 dt) override;

        void mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& colloc, View<f32, MultiSurface>& normals, View<f32, MultiSurface>& areas) override;
        f32 mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) override;
        f32 mesh_area(const View<f32, SingleSurface>& areas) override;
};

class BLAS_CUDA final : public BLAS {
    public:
        BLAS_CUDA() = default;
        ~BLAS_CUDA() = default;

        void gemv(const f32 alpha, const View<f32, Tensor<2>>& A, const View<f32, Tensor<1>>& x, const f32 beta, View<f32, Tensor<1>>& y, Trans trans = Trans::No) override;
        void gemm(const f32 alpha, const View<f32, Tensor<2>>& A, const View<f32, Tensor<2>>& B, const f32 beta, View<f32, Tensor<2>>& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) override;
};

class LUSolver_CUDA final : public LUSolver {
    public:
        LUSolver_CUDA() = default;
        ~LUSolver_CUDA() = default;

        void factorize(const View<f32, Tensor<2>>& A) = 0;
        void solve(const View<f32, Tensor<2>>& A, const View<f32, Tensor<1>>& x) = 0;
    private:
        Buffer<f32, MemoryLocation::Device, Tensor<1>> ipiv;
        Buffer<f32, MemoryLocation::Device, Tensor<1>> buffer;
        f32* info = nullptr;
};

} // namespace vlm
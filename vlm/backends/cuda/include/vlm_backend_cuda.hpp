#pragma once 

#include "vlm_backend.hpp" // base class
#include "vlm_data.hpp" // TODO: remove ?
#include "vlm_memory.hpp" // View, Layouts

namespace vlm {

class BackendCUDA final : public Backend {
    public:
        BackendCUDA();
        ~BackendCUDA() override;
        void lhs_assemble(TensorView2fD& lhs, const MultiTensorView3fD& colloc, const MultiTensorView3fD& normals, const MultiTensorView3fD& verts_wing, const MultiTensorView3fD& verts_wake, std::vector<i32>& condition, i32 iteration) override;
        void rhs_assemble_velocities(TensorView1fD& rhs, const MultiTensorView3fD& normals, const MultiTensorView3fD& velocities) override;
        void rhs_assemble_wake_influence(TensorView1fD& rhs, const MultiTensorView2fD& gamma_wake, const MultiTensorView3fD& colloc, const MultiTensorView3fD& normals, const MultiTensorView3fD& verts_wake, const std::vector<bool>& lifting, i32 iteration) override;
        void displace_wake_rollup(MultiTensorView3fD& wake_rollup, const MultiTensorView3fD& verts_wake, const MultiTensorView3fD& verts_wing, const MultiTensorView2fD& gamma_wing, const MultiTensorView2fD& gamma_wake, f32 dt, i32 iteration) override;

        // TODO: deprecate
        f32 coeff_steady_cl_single(const TensorView3fD& verts_wing, const TensorView2fD& gamma_delta, const FlowData& flow, f32 area) override;
        f32 coeff_steady_cd_single(const TensorView3fD& verts_wake, const TensorView2fD& gamma_wake, const FlowData& flow, f32 area) override;

        void forces_unsteady(
            const TensorView3fD& verts_wing,
            const TensorView2fD& gamma_delta,
            const TensorView2fD& gamma,
            const TensorView2fD& gamma_prev,
            const TensorView3fD& velocities,
            const TensorView2fD& areas,
            const TensorView3fD& normals,
            const TensorView3fD& forces,
            f32 dt
        ) override;
        f32 coeff_cl(
            const TensorView3fD& forces,
            const linalg::float3& lift_axis,
            const linalg::float3& freestream,
            const f32 rho,
            const f32 area
        ) override;
        linalg::float3 coeff_cm(
            const TensorView3fD& forces,
            const TensorView3fD& verts_wing,
            const linalg::float3& ref_pt,
            const linalg::float3& freestream,
            const f32 rho,
            const f32 area,
            const f32 mac
        ) override;

        void mesh_metrics(const f32 alpha_rad, const MultiTensorView3fD& verts_wing, MultiTensorView3fD& colloc, MultiTensorView3fD& normals, MultiTensorView2fD& areas) override;
        f32 mesh_mac(const TensorView3fD& verts_wing, const TensorView2fD& areas) override;

        void gamma_wake_from_coeffs(
            const TensorView2fD& gamma_wake,
            const TensorView2fD& gamma_coeffs,
            i32 harmonics,
            f32 tn,
            f32 omega,
            f32 dt,
            i64 iteration
        ) override;

        f32 sum(const TensorView1fD& tensor) override;
        f32 sum(const TensorView2fD& tensor) override;

        std::unique_ptr<Memory> create_memory_manager() override;
        // std::unique_ptr<Kernels> create_kernels() override;
        std::unique_ptr<LU> create_lu_solver() override;
        std::unique_ptr<BLAS> create_blas() override;
        std::unique_ptr<LSQ> create_lsq_solver() override;

        // Intermediate values for reduction
        // Still not certain if this is the best way
        // Currently it makes the functions that use these values not thread safe
        // So multiple instances of the same function cannot run in parallel
        f32* d_cl;
        f32* d_cd;
        f32* d_cm_x;
        f32* d_cm_y;
        f32* d_cm_z;
        f32* d_mac;
};

} // namespace vlm
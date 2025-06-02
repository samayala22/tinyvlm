#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendCPU final : public Backend {
    public:
        BackendCPU();
        ~BackendCPU() override;
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
        void forces_unsteady2(
            const TensorView3fD& verts_wing,
            const TensorView2fD& gamma_delta, // chordwise delta
            const TensorView2fD& dgamma_dt, // dgamma/dt
            const TensorView3fD& velocities,
            const TensorView2fD& areas,
            const TensorView3fD& normals,
            const TensorView3fD& forces
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
        
        // DOUBLE PRECISION
        void lhs_assemble(TensorView2dD& lhs, const MultiTensorView3dD& colloc, const MultiTensorView3dD& normals, const MultiTensorView3dD&  verts_wing, const MultiTensorView3dD&  verts_wake, std::vector<i32>& condition, i32 iteration) override;
        void rhs_assemble_velocities(TensorView1dD& rhs, const MultiTensorView3dD& normals, const MultiTensorView3dD& velocities) override;
        void rhs_assemble_wake_influence(TensorView1dD& rhs, const MultiTensorView2dD& gamma_wake, const MultiTensorView3dD& colloc, const MultiTensorView3dD& normals, const MultiTensorView3dD&  verts_wake, const std::vector<bool>& lifting, i32 iteration) override;
        void displace_wing(const MultiTensorView2dD& transforms, MultiTensorView3dD&  verts_wing, MultiTensorView3dD& verts_wing_init);
        void wake_shed(const MultiTensorView3dD& verts_wing, MultiTensorView3dD& verts_wake, i32 iteration);

        void forces_unsteady2(
            const TensorView3dD& verts_wing,
            const TensorView2dD& gamma_delta, // chordwise delta
            const TensorView2dD& dgamma_dt, // dgamma/dt
            const TensorView3dD& velocities,
            const TensorView2dD& areas,
            const TensorView3dD& normals,
            const TensorView3dD& forces
        ) override;
        f64 coeff_cl(
            const TensorView3dD& forces,
            const linalg::double3& lift_axis,
            const linalg::double3& freestream,
            const f64 rho,
            const f64 area
        ) override;
        linalg::double3 coeff_cm(
            const TensorView3dD& forces,
            const TensorView3dD& verts_wing,
            const linalg::double3& ref_pt,
            const linalg::double3& freestream,
            const f64 rho,
            const f64 area,
            const f64 mac
        ) override;

        void forces_unsteady_multibody(
            const MultiTensorView3dD& verts_wing,
            const MultiTensorView2dD& gamma_delta,
            const MultiTensorView2dD& gamma,
            const MultiTensorView2dD& gamma_prev,
            const MultiTensorView3dD& velocities,
            const MultiTensorView2dD& areas,
            const MultiTensorView3dD& normals,
            const MultiTensorView3dD& forces,
            f64 dt
        );
        f64 coeff_cl_multibody(
            const MultiTensorView3dD& aero_forces,
            const MultiTensorView2dD& areas,
            const linalg::double3& freestream,
            f64 rho
        );
        linalg::double3 coeff_cm_multibody(
            const MultiTensorView3dD& aero_forces,
            const MultiTensorView3dD& verts_wing,
            const MultiTensorView2dD& areas,
            const linalg::double3& ref_pt,
            const linalg::double3& freestream, 
            f64 rho
        );

        void mesh_metrics(const f64 alpha_rad, const MultiTensorView3dD&  verts_wing, MultiTensorView3dD& colloc, MultiTensorView3dD& normals, MultiTensorView2dD& areas) override;
        f64 mesh_mac(const TensorView3dD& verts_wing, const TensorView2dD& areas) override;
        void gamma_wake_from_coeffs(
            const TensorView2dD& gamma_wake,
            const TensorView2dD& gamma_coeffs,
            i32 harmonics,
            f64 tn,
            f64 omega,
            f64 dt,
            i64 iteration
        ) override;
        f64 sum(const TensorView1dD& tensor) override;
        f64 sum(const TensorView2dD& tensor) override;

        std::unique_ptr<Memory> create_memory_manager() override;
        // std::unique_ptr<Kernels> create_kernels() override;
        std::unique_ptr<LU> create_lu_solver() override;
        std::unique_ptr<BLAS> create_blas() override;
        std::unique_ptr<LSQ> create_lsq_solver() override;
};

} // namespace vlm
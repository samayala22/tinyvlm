#pragma once 

#include "vlm_backend.hpp" // base class
#include "vlm_memory.hpp" // View, Layouts

namespace vlm {

class BackendCUDA final : public Backend {
    public:
        BackendCUDA();
        ~BackendCUDA() override;

        void lhs_assemble(TV2f& lhs, MTV3f& colloc, MTV3f& normals, MTV3f& verts_wing, MTV3f& verts_wake, std::vector<i32>& condition, i32 iteration) override;
        void rhs_assemble_velocities(TV1f& rhs, MTV3f& normals, MTV3f& velocities) override;
        void rhs_assemble_wake_influence(TV1f& rhs, MTV2f& gamma_wake, MTV3f& colloc, MTV3f& normals, MTV3f& verts_wake, std::vector<bool>& lifting, i32 iteration) override;
        void forces_steady(TV3f& verts_wing, TV2f& gamma_delta, TV3f& velocities, TV3f& forces) override;
        void forces_unsteady(TV3f& verts_wing, TV2f& gamma_delta, TV2f& dgamma_dt, TV3f& velocities, TV2f& areas, TV3f& normals, TV3f& forces) override;
        f32 coeff_cl(TV3f& forces, f32x3& lift_axis, f32x3& freestream, f32 rho, f32 area) override;
        f32x3 coeff_cm(TV3f& forces, TV3f& verts_wing, f32x3& ref_pt, f32x3& freestream, f32 rho, f32 area, f32 mac) override;
        void mesh_metrics(f32 alpha_rad, MTV3f& verts_wing, MTV3f& colloc, MTV3f& normals, MTV2f& areas) override;
        f32 mesh_mac(TV3f& verts_wing, TV2f& areas) override;
        void gamma_wake_from_coeffs(TV2f& gamma_wake, TV2f& gamma_coeffs, i32 harmonics, f32 tn, f32 omega, f32 dt, i64 iteration) override;
        f32 sum(TV1f& tensor) override;
        f32 sum(TV2f& tensor) override;

        void lhs_assemble(TV2d& lhs, MTV3d& colloc, MTV3d& normals, MTV3d& verts_wing, MTV3d& verts_wake, std::vector<i32>& condition, i32 iteration) override;
        void rhs_assemble_velocities(TV1d& rhs, MTV3d& normals, MTV3d& velocities) override;
        void rhs_assemble_wake_influence(TV1d& rhs, MTV2d& gamma_wake, MTV3d& colloc, MTV3d& normals, MTV3d& verts_wake, std::vector<bool>& lifting, i32 iteration) override;
        void forces_steady(TV3d& verts_wing, TV2d& gamma_delta, TV3d& velocities, TV3d& forces) override;
        void forces_unsteady(TV3d& verts_wing, TV2d& gamma_delta, TV2d& dgamma_dt, TV3d& velocities, TV2d& areas, TV3d& normals, TV3d& forces) override;
        f64 coeff_cl(TV3d& forces, f64x3& lift_axis, f64x3& freestream, f64 rho, f64 area) override;
        f64x3 coeff_cm(TV3d& forces, TV3d& verts_wing, f64x3& ref_pt, f64x3& freestream, f64 rho, f64 area, f64 mac) override;
        void mesh_metrics(f64 alpha_rad, MTV3d& verts_wing, MTV3d& colloc, MTV3d& normals, MTV2d& areas) override;
        f64 mesh_mac(TV3d& verts_wing, TV2d& areas) override;
        void gamma_wake_from_coeffs(TV2d& gamma_wake, TV2d& gamma_coeffs, i32 harmonics, f64 tn, f64 omega, f64 dt, i64 iteration) override;
        f64 sum(TV1d& tensor) override;
        f64 sum(TV2d& tensor) override;

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
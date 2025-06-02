#pragma once

#include <memory>

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_data.hpp"
#include "vlm_memory.hpp"

#include "linalg.h"

namespace vlm {

class Backend {
    public:
        std::string name;
        const std::unique_ptr<Memory> memory;
        std::unique_ptr<BLAS> blas;

        f32 sigma_vatistas = 0.0f;
        Backend(std::unique_ptr<Memory> memory_, std::unique_ptr<BLAS> blas_) : memory(std::move(memory_)), blas(std::move(blas_)) {};
        virtual ~Backend() = default;

        // Kernels that run for all the meshes
        virtual void lhs_assemble(TensorView2fD& lhs, const MultiTensorView3fD& colloc, const MultiTensorView3fD& normals, const MultiTensorView3fD&  verts_wing, const MultiTensorView3fD&  verts_wake, std::vector<i32>& condition, i32 iteration) = 0;
        virtual void rhs_assemble_velocities(TensorView1fD& rhs, const MultiTensorView3fD& normals, const MultiTensorView3fD& velocities) = 0;
        virtual void rhs_assemble_wake_influence(TensorView1fD& rhs, const MultiTensorView2fD& gamma_wake, const MultiTensorView3fD& colloc, const MultiTensorView3fD& normals, const MultiTensorView3fD&  verts_wake, const std::vector<bool>& lifting, i32 iteration) = 0;
        virtual void displace_wake_rollup(MultiTensorView3fD& wake_rollup, const MultiTensorView3fD&  verts_wake, const MultiTensorView3fD&  verts_wing, const MultiTensorView2fD& gamma_wing, const MultiTensorView2fD& gamma_wake, f32 dt, i32 iteration) = 0;
        void displace_wing(const MultiTensorView2fD& transforms, MultiTensorView3fD&  verts_wing, MultiTensorView3fD& verts_wing_init);
        void wake_shed(const MultiTensorView3fD& verts_wing, MultiTensorView3fD& verts_wake, i32 iteration);

        [[deprecated]] virtual f32 coeff_steady_cl_single(const TensorView3fD& verts_wing, const TensorView2fD& gamma_delta, const FlowData& flow, f32 area) = 0;
        [[deprecated]] f32 coeff_steady_cl_multi(const MultiTensorView3fD&  verts_wing, const MultiTensorView2fD& gamma_delta, const FlowData& flow, const MultiTensorView2fD& areas);
        [[deprecated]] virtual f32 coeff_steady_cd_single(const TensorView3fD& verts_wake, const TensorView2fD& gamma_wake, const FlowData& flow, f32 area) = 0;
        [[deprecated]] f32 coeff_steady_cd_multi(const MultiTensorView3fD&  verts_wake, const MultiTensorView2fD& gamma_wake, const FlowData& flow, const MultiTensorView2fD& areas);
        
        virtual void forces_unsteady(
            const TensorView3fD& verts_wing,
            const TensorView2fD& gamma_delta,
            const TensorView2fD& gamma,
            const TensorView2fD& gamma_prev,
            const TensorView3fD& velocities,
            const TensorView2fD& areas,
            const TensorView3fD& normals,
            const TensorView3fD& forces,
            f32 dt
        ) = 0;
        virtual void forces_unsteady2(
            const TensorView3fD& verts_wing,
            const TensorView2fD& gamma_delta, // chordwise delta
            const TensorView2fD& dgamma_dt, // dgamma/dt
            const TensorView3fD& velocities,
            const TensorView2fD& areas,
            const TensorView3fD& normals,
            const TensorView3fD& forces
        ) = 0;
        virtual f32 coeff_cl(
            const TensorView3fD& forces,
            const linalg::float3& lift_axis,
            const linalg::float3& freestream,
            const f32 rho,
            const f32 area
        ) = 0;
        virtual linalg::float3 coeff_cm(
            const TensorView3fD& forces,
            const TensorView3fD& verts_wing,
            const linalg::float3& ref_pt,
            const linalg::float3& freestream,
            const f32 rho,
            const f32 area,
            const f32 mac
        ) = 0;

        void forces_unsteady_multibody(
            const MultiTensorView3fD& verts_wing,
            const MultiTensorView2fD& gamma_delta,
            const MultiTensorView2fD& gamma,
            const MultiTensorView2fD& gamma_prev,
            const MultiTensorView3fD& velocities,
            const MultiTensorView2fD& areas,
            const MultiTensorView3fD& normals,
            const MultiTensorView3fD& forces,
            f32 dt
        );
        f32 coeff_cl_multibody(
            const MultiTensorView3fD& aero_forces,
            const MultiTensorView2fD& areas,
            const linalg::float3& freestream,
            f32 rho
        );
        linalg::float3 coeff_cm_multibody(
            const MultiTensorView3fD& aero_forces,
            const MultiTensorView3fD& verts_wing,
            const MultiTensorView2fD& areas,
            const linalg::float3& ref_pt,
            const linalg::float3& freestream, 
            f32 rho
        );

        virtual void mesh_metrics(const f32 alpha_rad, const MultiTensorView3fD&  verts_wing, MultiTensorView3fD& colloc, MultiTensorView3fD& normals, MultiTensorView2fD& areas) = 0;
        virtual f32 mesh_mac(const TensorView3fD& verts_wing, const TensorView2fD& areas) = 0;
        virtual void gamma_wake_from_coeffs(
            const TensorView2fD& gamma_wake,
            const TensorView2fD& gamma_coeffs,
            i32 harmonics,
            f32 tn,
            f32 omega,
            f32 dt,
            i64 iteration
        ) = 0;
        virtual f32 sum(const TensorView1fD& tensor) = 0;
        virtual f32 sum(const TensorView2fD& tensor) = 0;

        // DOUBLE PRECISION
        virtual void lhs_assemble(TensorView2dD& lhs, const MultiTensorView3dD& colloc, const MultiTensorView3dD& normals, const MultiTensorView3dD&  verts_wing, const MultiTensorView3dD&  verts_wake, std::vector<i32>& condition, i32 iteration) = 0;
        virtual void rhs_assemble_velocities(TensorView1dD& rhs, const MultiTensorView3dD& normals, const MultiTensorView3dD& velocities) = 0;
        virtual void rhs_assemble_wake_influence(TensorView1dD& rhs, const MultiTensorView2dD& gamma_wake, const MultiTensorView3dD& colloc, const MultiTensorView3dD& normals, const MultiTensorView3dD&  verts_wake, const std::vector<bool>& lifting, i32 iteration) = 0;
        void displace_wing(const MultiTensorView2dD& transforms, MultiTensorView3dD&  verts_wing, MultiTensorView3dD& verts_wing_init);
        void wake_shed(const MultiTensorView3dD& verts_wing, MultiTensorView3dD& verts_wake, i32 iteration);

        virtual void forces_unsteady2(
            const TensorView3dD& verts_wing,
            const TensorView2dD& gamma_delta, // chordwise delta
            const TensorView2dD& dgamma_dt, // dgamma/dt
            const TensorView3dD& velocities,
            const TensorView2dD& areas,
            const TensorView3dD& normals,
            const TensorView3dD& forces
        ) = 0;
        virtual f64 coeff_cl(
            const TensorView3dD& forces,
            const linalg::double3& lift_axis,
            const linalg::double3& freestream,
            const f64 rho,
            const f64 area
        ) = 0;
        virtual linalg::double3 coeff_cm(
            const TensorView3dD& forces,
            const TensorView3dD& verts_wing,
            const linalg::double3& ref_pt,
            const linalg::double3& freestream,
            const f64 rho,
            const f64 area,
            const f64 mac
        ) = 0;

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

        virtual void mesh_metrics(const f64 alpha_rad, const MultiTensorView3dD&  verts_wing, MultiTensorView3dD& colloc, MultiTensorView3dD& normals, MultiTensorView2dD& areas) = 0;
        virtual f64 mesh_mac(const TensorView3dD& verts_wing, const TensorView2dD& areas) = 0;
        virtual void gamma_wake_from_coeffs(
            const TensorView2dD& gamma_wake,
            const TensorView2dD& gamma_coeffs,
            i32 harmonics,
            f64 tn,
            f64 omega,
            f64 dt,
            i64 iteration
        ) = 0;
        virtual f64 sum(const TensorView1dD& tensor) = 0;
        virtual f64 sum(const TensorView2dD& tensor) = 0;

        virtual std::unique_ptr<Memory> create_memory_manager() = 0;
        virtual std::unique_ptr<LU> create_lu_solver() = 0;
        virtual std::unique_ptr<BLAS> create_blas() = 0;
        virtual std::unique_ptr<LSQ> create_lsq_solver() = 0;
};

class BLAS {
    public:
        enum class Trans { Yes, No };
        explicit BLAS() = default;
        virtual ~BLAS() = default;

        virtual void gemv(const f32 alpha, const TensorView2fD& A, const TensorView1fD& x, const f32 beta, const TensorView1fD& y, Trans trans = Trans::No) = 0;
        virtual void gemm(const f32 alpha, const TensorView2fD& A, const TensorView2fD& B, const f32 beta, const TensorView2fD& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) = 0;
        virtual void axpy(const f32 alpha, const TensorView1fD& x, const TensorView1fD& y) = 0;
        virtual void axpy(const f32 alpha, const TensorView2fD& x, const TensorView2fD& y) = 0; // Y = alpha * X + Y
        virtual void scal(const f32 alpha, const TensorView1fD& x) = 0;
        virtual f32 norm(const TensorView1fD& x) = 0;

        virtual void gemv(const f64 alpha, const TensorView2dD& A, const TensorView1dD& x, const f64 beta, const TensorView1dD& y, Trans trans = Trans::No) = 0;
        virtual void gemm(const f64 alpha, const TensorView2dD& A, const TensorView2dD& B, const f64 beta, const TensorView2dD& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) = 0;
        virtual void axpy(const f64 alpha, const TensorView1dD& x, const TensorView1dD& y) = 0;
        virtual void axpy(const f64 alpha, const TensorView2dD& x, const TensorView2dD& y) = 0; // Y = alpha * X + Y
        virtual void scal(const f64 alpha, const TensorView1dD& x) = 0;
        virtual f64 norm(const TensorView1dD& x) = 0;
};

class LU {
    public:
        explicit LU(std::unique_ptr<Memory> memory) : m_memory(std::move(memory)) {}
        virtual ~LU() = default;
        
        virtual void init(const TensorView2fD& A) = 0;
        virtual void factorize(const TensorView2fD& A) = 0;
        virtual void solve(const TensorView2fD& A, const TensorView2fD& x) = 0;
        void solve(const TensorView2fD& A, const TensorView1fD& x) {
            return solve(A, x.reshape(x.shape(0), 1));
        }

        virtual void init(const TensorView2dD& A) = 0;
        virtual void factorize(const TensorView2dD& A) = 0;
        virtual void solve(const TensorView2dD& A, const TensorView2dD& x) = 0;
        void solve(const TensorView2dD& A, const TensorView1dD& x) {
            return solve(A, x.reshape(x.shape(0), 1));
        }
    protected:
        std::unique_ptr<Memory> m_memory;
};

class LSQ {
    public:
        explicit LSQ(std::unique_ptr<Memory> memory) : m_memory(std::move(memory)) {}
        virtual ~LSQ() = default;
        
        virtual void init(const TensorView2fD& A, const TensorView2fD& B) = 0;
        virtual void solve(const TensorView2fD& A, const TensorView2fD& B) = 0;
        virtual void init(const TensorView2dD& A, const TensorView2dD& B) = 0;
        virtual void solve(const TensorView2dD& A, const TensorView2dD& B) = 0;

    protected:
        std::unique_ptr<Memory> m_memory;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name);
std::vector<std::string> get_available_backends();

} // namespace vlm

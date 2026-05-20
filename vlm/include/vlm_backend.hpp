#pragma once

#include <memory>

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_memory.hpp"

#include "linalg.h"

namespace vlm {

class Backend {
public:
    // Short aliases to keep declarations on a single line
    using TV1f  = TensorView1fD;  using TV2f  = TensorView2fD;  using TV3f  = TensorView3fD;
    using MTV2f = MultiTensorView2fD; using MTV3f = MultiTensorView3fD;
    using TV1d  = TensorView1dD;  using TV2d  = TensorView2dD;  using TV3d  = TensorView3dD;
    using MTV2d = MultiTensorView2dD; using MTV3d = MultiTensorView3dD;
    using f32x3 = linalg::float3; using f64x3 = linalg::double3;

    std::string name;
    std::unique_ptr<Memory> memory;
    std::unique_ptr<BLAS> blas;
    f32 sigma_vatistas = 0.0f;

    Backend(std::unique_ptr<Memory> memory_, std::unique_ptr<BLAS> blas_) : memory(std::move(memory_)), blas(std::move(blas_)) {}
    virtual ~Backend() = default;

    // Pure Virtual functions
    virtual void lhs_assemble(TV2f& lhs, MTV3f& colloc, MTV3f& normals, MTV3f& verts_wing, MTV3f& verts_wake, std::vector<i32>& condition, i32 iteration) = 0;
    virtual void rhs_assemble_velocities(TV1f& rhs, MTV3f& normals, MTV3f& velocities) = 0;
    virtual void rhs_assemble_wake_influence(TV1f& rhs, MTV2f& gamma_wake, MTV3f& colloc, MTV3f& normals, MTV3f& verts_wake, std::vector<bool>& lifting, i32 iteration) = 0;
    virtual void forces_steady(TV3f& verts_wing, TV2f& gamma_delta, TV3f& velocities, TV3f& forces) = 0;
    virtual void forces_unsteady(TV3f& verts_wing, TV2f& gamma_delta, TV2f& dgamma_dt, TV3f& velocities, TV2f& areas, TV3f& normals, TV3f& forces) = 0;
    virtual f32 coeff_cl(TV3f& forces, f32x3& lift_axis, f32x3& freestream, f32 rho, f32 area) = 0;
    virtual f32x3 coeff_cm(TV3f& forces, TV3f& verts_wing, f32x3& ref_pt, f32x3& freestream, f32 rho, f32 area, f32 mac) = 0;
    virtual void mesh_metrics(f32 alpha_rad, MTV3f& verts_wing, MTV3f& colloc, MTV3f& normals, MTV2f& areas) = 0;
    virtual f32 mesh_mac(TV3f& verts_wing, TV2f& areas) = 0;
    virtual void gamma_wake_from_coeffs(TV2f& gamma_wake, TV2f& gamma_coeffs, i32 harmonics, f32 tn, f32 omega, f32 dt, i64 iteration) = 0;
    virtual f32 sum(TV1f& tensor) = 0;
    virtual f32 sum(TV2f& tensor) = 0;

    virtual void lhs_assemble(TV2d& lhs, MTV3d& colloc, MTV3d& normals, MTV3d& verts_wing, MTV3d& verts_wake, std::vector<i32>& condition, i32 iteration) = 0;
    virtual void rhs_assemble_velocities(TV1d& rhs, MTV3d& normals, MTV3d& velocities) = 0;
    virtual void rhs_assemble_wake_influence(TV1d& rhs, MTV2d& gamma_wake, MTV3d& colloc, MTV3d& normals, MTV3d& verts_wake, std::vector<bool>& lifting, i32 iteration) = 0;
    virtual void forces_steady(TV3d& verts_wing, TV2d& gamma_delta, TV3d& velocities, TV3d& forces) = 0;
    virtual void forces_unsteady(TV3d& verts_wing, TV2d& gamma_delta, TV2d& dgamma_dt, TV3d& velocities, TV2d& areas, TV3d& normals, TV3d& forces) = 0;
    virtual f64 coeff_cl(TV3d& forces, f64x3& lift_axis, f64x3& freestream, f64 rho, f64 area) = 0;
    virtual f64x3 coeff_cm(TV3d& forces, TV3d& verts_wing, f64x3& ref_pt, f64x3& freestream, f64 rho, f64 area, f64 mac) = 0;
    virtual void mesh_metrics(f64 alpha_rad, MTV3d& verts_wing, MTV3d& colloc, MTV3d& normals, MTV2d& areas) = 0;
    virtual f64 mesh_mac(TV3d& verts_wing, TV2d& areas) = 0;
    virtual void gamma_wake_from_coeffs(TV2d& gamma_wake, TV2d& gamma_coeffs, i32 harmonics, f64 tn, f64 omega, f64 dt, i64 iteration) = 0;
    virtual f64 sum(TV1d& tensor) = 0;
    virtual f64 sum(TV2d& tensor) = 0;

    // Non-Virtual functions
    void displace_wing(MTV2f& transforms, MTV3f& verts_wing, MTV3f& verts_wing_init);
    void wake_shed(MTV3f& verts_wing, MTV3f& verts_wake, i32 iteration);
    void forces_steady_multibody(MTV3f& verts_wing, MTV2f& gamma_delta, MTV3f& velocities, MTV3f& forces);
    void forces_unsteady_multibody(MTV3f& verts_wing, MTV2f& gamma_delta, MTV2f& dgamma_dt, MTV3f& velocities, MTV2f& areas, MTV3f& normals, MTV3f& forces);
    f32 coeff_cl_multibody(MTV3f& aero_forces, MTV2f& areas, f32x3& freestream, f32 rho);
    f32x3 coeff_cm_multibody(MTV3f& aero_forces, MTV3f& verts_wing, MTV2f& areas, f32x3& ref_pt, f32x3& freestream, f32 rho);

    void displace_wing(MTV2d& transforms, MTV3d& verts_wing, MTV3d& verts_wing_init);
    void wake_shed(MTV3d& verts_wing, MTV3d& verts_wake, i32 iteration);
    void forces_steady_multibody(MTV3d& verts_wing, MTV2d& gamma_delta, MTV3d& velocities, MTV3d& forces);
    void forces_unsteady_multibody(MTV3d& verts_wing, MTV2d& gamma_delta, MTV2d& dgamma_dt, MTV3d& velocities, MTV2d& areas, MTV3d& normals, MTV3d& forces);
    f64 coeff_cl_multibody(MTV3d& aero_forces, MTV2d& areas, f64x3& freestream, f64 rho);
    f64x3 coeff_cm_multibody(MTV3d& aero_forces, MTV3d& verts_wing, MTV2d& areas, f64x3& ref_pt, f64x3& freestream, f64 rho);

    virtual std::unique_ptr<Memory> create_memory_manager() = 0;
    virtual std::unique_ptr<LU> create_lu_solver() = 0;
    virtual std::unique_ptr<BLAS> create_blas() = 0;
    virtual std::unique_ptr<LSQ> create_lsq_solver() = 0;
};

class BLAS {
    public:
        enum class Trans { Yes, No }; // TODO: replace with a boolean
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

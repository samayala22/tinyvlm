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
        virtual void lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3fD& colloc, const MultiTensorView3fD& normals, const MultiTensorView3fD&  verts_wing, const MultiTensorView3fD&  verts_wake, std::vector<i32>& condition, i32 iteration) = 0;
        virtual void rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3fD& normals, const MultiTensorView3fD& velocities) = 0;
        virtual void rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView2fD& gamma_wake, const MultiTensorView3fD& colloc, const MultiTensorView3fD& normals, const MultiTensorView3fD&  verts_wake, const std::vector<bool>& lifting, i32 iteration) = 0;
        virtual void displace_wake_rollup(MultiTensorView3fD& wake_rollup, const MultiTensorView3fD&  verts_wake, const MultiTensorView3fD&  verts_wing, const MultiTensorView2fD& gamma_wing, const MultiTensorView2fD& gamma_wake, f32 dt, i32 iteration) = 0;
        void displace_wing(const MultiTensorView2fD& transforms, MultiTensorView3fD&  verts_wing, MultiTensorView3fD& verts_wing_init);
        void wake_shed(const MultiTensorView3fD& verts_wing, MultiTensorView3fD& verts_wake, i32 iteration);

        // TODO: deprecate
        virtual f32 coeff_steady_cl_single(const TensorView3fD& verts_wing, const TensorView2fD& gamma_delta, const FlowData& flow, f32 area) = 0;
        f32 coeff_steady_cl_multi(const MultiTensorView3fD&  verts_wing, const MultiTensorView2fD& gamma_delta, const FlowData& flow, const MultiTensorView2fD& areas);
        virtual f32 coeff_steady_cd_single(const TensorView3fD& verts_wake, const TensorView2fD& gamma_wake, const FlowData& flow, f32 area) = 0;
        f32 coeff_steady_cd_multi(const MultiTensorView3fD&  verts_wake, const MultiTensorView2fD& gamma_wake, const FlowData& flow, const MultiTensorView2fD& areas);
        
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

        virtual f32 sum(const TensorView1fD& tensor) = 0;
        virtual f32 sum(const TensorView2fD& tensor) = 0;

        virtual std::unique_ptr<Memory> create_memory_manager() = 0; // todo: deprecate
        // virtual std::unique_ptr<Kernels> create_kernels() = 0;
        virtual std::unique_ptr<LU> create_lu_solver() = 0;
        virtual std::unique_ptr<BLAS> create_blas() = 0; // todo: deprecate
        virtual std::unique_ptr<LSQ> create_lsq() = 0;
};

class BLAS {
    public:
        enum class Trans { Yes, No };
        explicit BLAS() = default;
        virtual ~BLAS() = default;

        virtual void gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, const TensorView<f32, 1, Location::Device>& y, Trans trans = Trans::No) = 0;
        virtual void gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, const TensorView<f32, 2, Location::Device>& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) = 0;
        virtual void axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, const TensorView<f32, 1, Location::Device>& y) = 0;
        virtual void axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, const TensorView<f32, 2, Location::Device>& y) = 0; // Y = alpha * X + Y
        virtual f32 norm(const TensorView<f32, 1, Location::Device>& x) = 0;
};

class LU {
    public:
        explicit LU(std::unique_ptr<Memory> memory) : m_memory(std::move(memory)) {}
        virtual ~LU() = default;
        
        virtual void init(const TensorView<f32, 2, Location::Device>& A) = 0;
        virtual void factorize(const TensorView<f32, 2, Location::Device>& A) = 0;
        virtual void solve(const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& x) = 0;
        void solve(const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x) {
            return solve(A, x.reshape(x.shape(0), 1));
        }
    protected:
        std::unique_ptr<Memory> m_memory;
};

class LSQ {
    public:
        explicit LSQ(std::unique_ptr<Memory> memory) : m_memory(std::move(memory)) {}
        virtual ~LSQ() = default;
        
        virtual void solve(
            const TensorView<f32, 2, Location::Device>& A,
            const TensorView<f32, 2, Location::Device>& B
        ) = 0;

    protected:
        std::unique_ptr<Memory> m_memory;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name);
std::vector<std::string> get_available_backends();

} // namespace vlm

#pragma once

#include <memory>

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_memory.hpp"

#include "linalg.h"

namespace vlm {

class Backend {
    public:
        const std::unique_ptr<Memory> memory;
        std::unique_ptr<BLAS> blas;

        // TODO: remove these
        i32* d_solver_info = nullptr;
        i32* d_solver_ipiv = nullptr;
        f32* d_solver_buffer = nullptr;
        f32* d_val = nullptr; // intermediate value used for reduction

        f32 sigma_vatistas = 0.0f;
        Backend(std::unique_ptr<Memory> memory_, std::unique_ptr<BLAS> blas_);
        virtual ~Backend();

        // Kernels that run for all the meshes
        virtual void lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>&  verts_wing, const MultiTensorView3D<Location::Device>&  verts_wake, std::vector<i32>& condition, i32 iteration) = 0;
        virtual void rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& velocities) = 0;
        virtual void rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView2D<Location::Device>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>&  verts_wake, i32 iteration) = 0;
        virtual void displace_wake_rollup(MultiTensorView3D<Location::Device>& wake_rollup, const MultiTensorView3D<Location::Device>&  verts_wake, const MultiTensorView3D<Location::Device>&  verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wake, f32 dt, i32 iteration) = 0;
        void displace_wing(const MultiTensorView2D<Location::Device>& transforms, MultiTensorView3D<Location::Device>&  verts_wing, MultiTensorView3D<Location::Device>& verts_wing_init);
        void wake_shed(const MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& verts_wake, i32 iteration);

        // Per mesh kernels
        virtual f32 coeff_steady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const FlowData& flow, f32 area) = 0;
        f32 coeff_steady_cl_multi(const MultiTensorView3D<Location::Device>&  verts_wing, const MultiTensorView2D<Location::Device>& gamma_delta, const FlowData& flow, const MultiTensorView2D<Location::Device>& areas);
        virtual f32 coeff_steady_cd_single(const TensorView3D<Location::Device>& verts_wake, const TensorView2D<Location::Device>& gamma_wake, const FlowData& flow, f32 area) = 0;
        f32 coeff_steady_cd_multi(const MultiTensorView3D<Location::Device>&  verts_wake, const MultiTensorView2D<Location::Device>& gamma_wake, const FlowData& flow, const MultiTensorView2D<Location::Device>& areas);
        virtual f32 coeff_unsteady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const TensorView2D<Location::Device>& gamma, const TensorView2D<Location::Device>& gamma_prev, const TensorView3D<Location::Device>& local_velocities, const TensorView2D<Location::Device>& areas, const TensorView3D<Location::Device>& normals, const linalg::float3& freestream, f32 dt, f32 area) = 0;
        f32 coeff_unsteady_cl_multi(const MultiTensorView3D<Location::Device>&  verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing_delta, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wing_prev, const MultiTensorView3D<Location::Device>& velocities, const MultiTensorView2D<Location::Device>& areas, const MultiTensorView3D<Location::Device>& normals, const linalg::float3& freestream, f32 dt);
        
        virtual void forces_unsteady(
            const TensorView3D<Location::Device>& verts_wing,
            const TensorView2D<Location::Device>& gamma_delta,
            const TensorView2D<Location::Device>& gamma,
            const TensorView2D<Location::Device>& gamma_prev,
            const TensorView3D<Location::Device>& velocities,
            const TensorView2D<Location::Device>& areas,
            const TensorView3D<Location::Device>& normals,
            const TensorView3D<Location::Device>& forces,
            f32 dt
        ) = 0;
        virtual f32 coeff_cl(
            const TensorView3D<Location::Device>& forces,
            const linalg::float3& lift_axis,
            const linalg::float3& freestream,
            const f32 rho,
            const f32 area
        ) = 0;
        virtual linalg::float3 coeff_cm(
            const TensorView3D<Location::Device>& forces,
            const TensorView3D<Location::Device>& verts_wing,
            const linalg::float3& ref_pt,
            const linalg::float3& freestream,
            const f32 rho,
            const f32 area,
            const f32 mac
        ) = 0;

        virtual void mesh_metrics(const f32 alpha_rad, const MultiTensorView3D<Location::Device>&  verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, MultiTensorView2D<Location::Device>& areas) = 0;
        virtual f32 mesh_mac(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& areas) = 0;

        virtual f32 sum(const TensorView1D<Location::Device>& tensor) = 0;
        virtual f32 sum(const TensorView2D<Location::Device>& tensor) = 0;

        virtual std::unique_ptr<Memory> create_memory_manager() = 0; // todo: deprecate
        // virtual std::unique_ptr<Kernels> create_kernels() = 0;
        virtual std::unique_ptr<LU> create_lu_solver() = 0;
        virtual std::unique_ptr<BLAS> create_blas() = 0; // todo: deprecate
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

std::unique_ptr<Backend> create_backend(const std::string& backend_name);
std::vector<std::string> get_available_backends();

} // namespace vlm

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
        Backend(std::unique_ptr<Memory> memory_) : memory(std::move(memory_)) {
            d_val = (f32*)memory->alloc(Location::Device, sizeof(f32));
        }
        virtual ~Backend();

        // Kernels that run for all the meshes
        virtual void lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<i32>& condition, i32 iteration) = 0;
        virtual void rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& velocities) = 0;
        virtual void rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const View<f32, MultiSurface>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wake, i32 iteration) = 0;
        virtual void displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, i32 iteration) = 0;
        virtual void displace_wing(const TensorView<f32, 3, Location::Device>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) = 0;
        virtual void wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, i32 iteration) = 0;

        virtual void gamma_shed(View<f32, MultiSurface>& gamma_wing, View<f32, MultiSurface>& gamma_wing_prev, View<f32, MultiSurface>& gamma_wake, i32 iteration) = 0;
        virtual void gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) = 0;
        
        // Per mesh kernels
        virtual f32 coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow, f32 area) = 0;
        f32 coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow, const View<f32, MultiSurface>& areas);
        virtual f32 coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const View<f32, SingleSurface>& gamma_wake, const FlowData& flow, f32 area) = 0;
        f32 coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& gamma_wake, const FlowData& flow, const View<f32, MultiSurface>& areas);
        virtual f32 coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& local_velocities, const View<f32, SingleSurface>& areas, const TensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) = 0;
        f32 coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const MultiTensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt);
        virtual void coeff_unsteady_cl_single_forces(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& velocities, const View<f32, SingleSurface>& areas, const TensorView3D<Location::Device>& normals, View<f32, SingleSurface>& forces, const linalg::alias::float3& freestream, f32 dt) = 0;
        void coeff_unsteady_cl_multi_forces(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& forces, const linalg::alias::float3& freestream, f32 dt);

        virtual void mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& areas) = 0;
        virtual f32 mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) = 0;
        virtual f32 mesh_area(const View<f32, SingleSurface>& areas) = 0;

        virtual std::unique_ptr<Memory> create_memory_manager() = 0; // todo: deprecate
        // virtual std::unique_ptr<Kernels> create_kernels() = 0;
        virtual std::unique_ptr<LU> create_lu_solver() = 0;
        virtual std::unique_ptr<BLAS> create_blas() = 0; // todo: deprecate
};

class BLAS {
    public:
        enum class Trans { Yes, No };
        BLAS() = default;
        virtual ~BLAS() = default;

        virtual void gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, TensorView<f32, 1, Location::Device>& y, Trans trans = Trans::No) = 0;
        virtual void gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, TensorView<f32, 2, Location::Device>& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) = 0;
        virtual void axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, TensorView<f32, 1, Location::Device>& y) = 0;
        virtual void axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, TensorView<f32, 2, Location::Device>& y) = 0;
        // virtual void xypz(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, const TensorView<f32, 1, Location::Device>& y, const f32 beta, TensorView<f32, 1, Location::Device>& z) = 0;
};

class LU {
    public:
        LU(std::unique_ptr<Memory> memory) : m_memory(std::move(memory)) {}
        virtual ~LU() = default;
        
        virtual void init(const TensorView<f32, 2, Location::Device>& A) = 0;
        virtual void factorize(const TensorView<f32, 2, Location::Device>& A) = 0;
        virtual void solve(const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& x) = 0;

    protected:
        std::unique_ptr<Memory> m_memory;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name);
std::vector<std::string> get_available_backends();

} // namespace vlm

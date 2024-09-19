#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_memory.hpp"

namespace vlm {

class Backend {
    public:
        const std::unique_ptr<Memory> memory;

        i32* d_solver_info = nullptr;
        i32* d_solver_ipiv = nullptr;
        f32* d_solver_buffer = nullptr;
        f32* d_val = nullptr; // intermediate value used for reduction

        f32 sigma_vatistas = 0.0f;
        Backend(std::unique_ptr<Memory> memory_) : memory(std::move(memory_)) {
            d_val = (f32*)memory->alloc(MemoryLocation::Device, sizeof(f32));
        }
        virtual ~Backend();

        // Kernels that run for all the meshes
        virtual void lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<u32>& condition, u32 iteration) = 0;
        virtual void rhs_assemble_velocities(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& velocities) = 0;
        virtual void rhs_assemble_wake_influence(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& gamma_wake, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wake, u32 iteration) = 0;
        virtual void displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, u32 iteration) = 0;
        virtual void displace_wing(const View<f32, Tensor<3>>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) = 0;
        virtual void wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, u32 iteration) = 0;
        virtual void gamma_shed(View<f32, MultiSurface>& gamma_wing, View<f32, MultiSurface>& gamma_wing_prev, View<f32, MultiSurface>& gamma_wake, u32 iteration) = 0;
        virtual void gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) = 0;
        virtual void lu_allocate(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) = 0;
        virtual void lu_factor(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) = 0;
        virtual void lu_solve(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, View<f32, MultiSurface>& rhs, View<f32, MultiSurface>& gamma) = 0;
        
        // Per mesh kernels 
        virtual f32 coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow, f32 area) = 0;
        virtual f32 coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow, const View<f32, MultiSurface>& areas) = 0;
        virtual f32 coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const View<f32, SingleSurface>& gamma_wake, const FlowData& flow, f32 area) = 0;
        virtual f32 coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& gamma_wake, const FlowData& flow, const View<f32, MultiSurface>& areas) = 0;
        virtual f32 coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& local_velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) = 0;
        virtual f32 coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const View<f32, MultiSurface>& normals, const linalg::alias::float3& freestream, f32 dt) = 0;
        // virtual linalg::alias::float3 coeff_steady_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) = 0;

        virtual void mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& colloc, View<f32, MultiSurface>& normals, View<f32, MultiSurface>& areas) = 0;
        virtual f32 mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) = 0;
        virtual f32 mesh_area(const View<f32, SingleSurface>& areas) = 0;
};

class BLAS {
    public:
        enum class Trans { Yes, No };
        BLAS() = default;
        virtual ~BLAS() = default;

        virtual void gemv(const f32 alpha, const View<f32, Tensor<2>>& A, const View<f32, Tensor<1>>& x, const f32 beta, View<f32, Tensor<1>>& y, Trans order = Trans::No) = 0;
        virtual void gemm(const f32 alpha, const View<f32, Tensor<2>>& A, const View<f32, Tensor<2>>& B, const f32 beta, View<f32, Tensor<2>>& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) = 0;
        virtual void getrf(const View<f32, Tensor<2>>& A, const View<i32, Tensor<1>>& ipiv) = 0;
        virtual void getrs(const View<f32, Tensor<2>>& A, const View<i32, Tensor<1>>& ipiv, const View<f32, Tensor<1>>& b) = 0;
};

class LUSolver {
    public:
        LUSolver() = default;
        virtual ~LUSolver() = default;

        virtual void factorize(const View<f32, Tensor<2>>& A) = 0;
        virtual void solve(const View<f32, Tensor<2>>& A, const View<f32, Tensor<1>>& x) = 0;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name);
std::vector<std::string> get_available_backends();

} // namespace vlm

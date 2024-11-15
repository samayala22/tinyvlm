#pragma once 

#include "vlm_backend.hpp" // base class
#include "vlm_data.hpp" // TODO: remove ?
#include "vlm_memory.hpp" // View, Layouts

namespace vlm {

class BackendCUDA final : public Backend {
    public:
        BackendCUDA();
        ~BackendCUDA() override;
        void lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView3D<Location::Device>& verts_wake, std::vector<i32>& condition, i32 iteration) override;
        void rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& velocities) override;
        void rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView2D<Location::Device>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& verts_wake, i32 iteration) override;
        void displace_wake_rollup(MultiTensorView3D<Location::Device>& wake_rollup, const MultiTensorView3D<Location::Device>& verts_wake, const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wake, f32 dt, i32 iteration) override;

        // Per mesh kernels
        f32 coeff_steady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const FlowData& flow, f32 area) override;
        f32 coeff_steady_cd_single(const TensorView3D<Location::Device>& verts_wake, const TensorView2D<Location::Device>& gamma_wake, const FlowData& flow, f32 area) override;
        f32 coeff_unsteady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const TensorView2D<Location::Device>& gamma, const TensorView2D<Location::Device>& gamma_prev, const TensorView3D<Location::Device>& local_velocities, const TensorView2D<Location::Device>& areas, const TensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) override;
        void coeff_unsteady_cl_single_forces(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const TensorView2D<Location::Device>& gamma, const TensorView2D<Location::Device>& gamma_prev, const TensorView3D<Location::Device>& velocities, const TensorView2D<Location::Device>& areas, const TensorView3D<Location::Device>& normals, TensorView3D<Location::Device>& forces, const linalg::alias::float3& freestream, f32 dt) override;

        void mesh_metrics(const f32 alpha_rad, const MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, MultiTensorView2D<Location::Device>& areas) override;
        f32 mesh_mac(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& areas) override;

        f32 sum(const TensorView1D<Location::Device>& tensor) override;
        f32 sum(const TensorView2D<Location::Device>& tensor) override;

        std::unique_ptr<Memory> create_memory_manager() override;
        // std::unique_ptr<Kernels> create_kernels() override;
        std::unique_ptr<LU> create_lu_solver() override;
        std::unique_ptr<BLAS> create_blas() override;
};

} // namespace vlm
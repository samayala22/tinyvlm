#pragma once 

#include "vlm_backend.hpp" // base class
#include "vlm_data.hpp" // TODO: remove ?
#include "vlm_memory.hpp" // View, Layouts

namespace vlm {

class BackendCUDA final : public Backend {
    public:
        BackendCUDA();
        ~BackendCUDA() override;
        void lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<i32>& condition, i32 iteration) override;
        void rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& velocities) override;
        void rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView2D<Location::Device>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wake, i32 iteration) override;
        void displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wake, f32 dt, i32 iteration) override;
        void displace_wing(const TensorView<f32, 3, Location::Device>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) override;
        void wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, i32 iteration) override;

        // Per mesh kernels
        f32 coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const FlowData& flow, f32 area) override;
        f32 coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const TensorView2D<Location::Device>& gamma_wake, const FlowData& flow, f32 area) override;
        f32 coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const TensorView2D<Location::Device>& gamma, const TensorView2D<Location::Device>& gamma_prev, const View<f32, SingleSurface>& local_velocities, const View<f32, SingleSurface>& areas, const TensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) override;
        void coeff_unsteady_cl_single_forces(const View<f32, SingleSurface>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const TensorView2D<Location::Device>& gamma, const TensorView2D<Location::Device>& gamma_prev, const View<f32, SingleSurface>& velocities, const View<f32, SingleSurface>& areas, const TensorView3D<Location::Device>& normals, View<f32, SingleSurface>& forces, const linalg::alias::float3& freestream, f32 dt) override;

        void mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& areas) override;
        f32 mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) override;
        f32 mesh_area(const View<f32, SingleSurface>& areas) override;

        std::unique_ptr<Memory> create_memory_manager() override;
        // std::unique_ptr<Kernels> create_kernels() override;
        std::unique_ptr<LU> create_lu_solver() override;
        std::unique_ptr<BLAS> create_blas() override;
};

} // namespace vlm
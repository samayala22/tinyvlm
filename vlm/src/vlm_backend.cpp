#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"

#include <string>

#ifdef VLM_CPU
#include "vlm_backend_cpu.hpp"
#endif
#ifdef VLM_CUDA
#include "vlm_backend_cuda.hpp"
#endif

using namespace vlm;

std::unique_ptr<Backend> vlm::create_backend(const std::string& backend_name) {
    #ifdef VLM_CPU
    if (backend_name == "cpu") {
        return std::make_unique<BackendCPU>();
    }
    #endif
    #ifdef VLM_CUDA
    if (backend_name == "cuda") {
        return std::make_unique<BackendCUDA>();
    }
    #endif
    throw std::runtime_error("Unsupported backend: " + backend_name); // TODO: remove
}

std::vector<std::string> vlm::get_available_backends() {
    std::vector<std::string> backends;
    #ifdef VLM_CPU
    backends.push_back("cpu");
    #endif
    #ifdef VLM_CUDA
    backends.push_back("cuda");
    #endif
    return backends;
}

Backend::Backend(std::unique_ptr<Memory> memory_, std::unique_ptr<BLAS> blas_) : memory(std::move(memory_)), blas(std::move(blas_)) {
    d_val = (f32*)memory->alloc(Location::Device, sizeof(f32));
}

Backend::~Backend() {
    // TODO: move this
    memory->free(Location::Device, d_solver_info); // deprecate
    memory->free(Location::Device, d_solver_ipiv); // deprecate
    memory->free(Location::Device, d_solver_buffer); // deprecate
    memory->free(Location::Device, d_val);
}

f32 Backend::coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_delta, const FlowData& flow, const MultiTensorView2D<Location::Device>& areas) {
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const f32 area_local = sum(areas[i]);
        const f32 wing_cl = coeff_steady_cl_single(
            verts_wing_local,
            gamma_delta[i],
            flow,
            area_local
        );
        cl += wing_cl * area_local;
        total_area += area_local;
    }
    cl /= total_area;
    return cl;
}

f32 Backend::coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing_delta, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const MultiTensorView2D<Location::Device>& areas, const MultiTensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt) {
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const auto velocities_local = velocities.layout.subview(velocities.ptr, i);

        const f32 area_local = sum(areas[i]);
        const f32 wing_cl = coeff_unsteady_cl_single(
            verts_wing_local,
            gamma_wing_delta[i],
            gamma_wing[i],
            gamma_wing_prev[i],
            velocities_local,
            areas[i],
            normals[i],
            freestream,
            dt,
            area_local
        );
        cl += wing_cl * area_local;
        total_area += area_local;
    }
    cl /= total_area;
    return cl;
}

void Backend::coeff_unsteady_cl_multi_forces(const View<f32, MultiSurface>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing_delta, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const MultiTensorView2D<Location::Device>& areas, const MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& forces, const linalg::alias::float3& freestream, f32 dt) {
    // todo: parallel
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const auto velocities_local = velocities.layout.subview(velocities.ptr, i);
        auto forces_local = forces.layout.subview(forces.ptr, i);
        coeff_unsteady_cl_single_forces(
            verts_wing_local, 
            gamma_wing_delta[i],
            gamma_wing[i],
            gamma_wing_prev[i],
            velocities_local,
            areas[i],
            normals[i],
            forces_local,
            freestream,
            dt
        );
    }
}

f32 Backend::coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const MultiTensorView2D<Location::Device>& gamma_wake, const FlowData& flow, const MultiTensorView2D<Location::Device>& areas) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cd = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wake.layout.surfaces().size(); i++) {
        const auto verts_wake_local = verts_wake.layout.subview(verts_wake.ptr, i);
        const f32 area_local = sum(areas[i]);
        const f32 wing_cd = coeff_steady_cd_single(
            verts_wake_local,
            gamma_wake[i],
            flow,
            area_local
        );
        cd += wing_cd * area_local;
        total_area += area_local;
    }
    cd /= total_area;
    return cd;
}
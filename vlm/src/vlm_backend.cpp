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

Backend::~Backend() {
    // Free device-device ptrs
    memory->free(Location::Device, d_solver_info);
    memory->free(Location::Device, d_solver_ipiv);
    memory->free(Location::Device, d_solver_buffer);
    memory->free(Location::Device, d_val);
}

f32 Backend::coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const auto gamma_delta_local = gamma_delta.layout.subview(gamma_delta.ptr, i);
        const auto areas_local = areas.layout.subview(areas.ptr, i);
        const f32 area_local = mesh_area(areas_local);
        const f32 wing_cl = coeff_steady_cl_single(verts_wing_local, gamma_delta_local, flow, area_local);
        cl += wing_cl * area_local;
        total_area += area_local;
    }
    cl /= total_area;
    return cl;
}

f32 Backend::coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const MultiTensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt) {
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const auto areas_local = areas.layout.subview(areas.ptr, i);
        const auto gamma_delta_local = gamma_wing_delta.layout.subview(gamma_wing_delta.ptr, i);
        const auto gamma_wing_local = gamma_wing.layout.subview(gamma_wing.ptr, i);
        const auto gamma_wing_prev_local = gamma_wing_prev.layout.subview(gamma_wing_prev.ptr, i);
        const auto velocities_local = velocities.layout.subview(velocities.ptr, i);

        const f32 area_local = mesh_area(areas_local);
        const f32 wing_cl = coeff_unsteady_cl_single(
            verts_wing_local,
            gamma_delta_local,
            gamma_wing_local,
            gamma_wing_prev_local,
            velocities_local,
            areas_local,
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

void Backend::coeff_unsteady_cl_multi_forces(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& forces, const linalg::alias::float3& freestream, f32 dt) {
    // todo: parallel
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const auto areas_local = areas.layout.subview(areas.ptr, i);
        const auto gamma_delta_local = gamma_wing_delta.layout.subview(gamma_wing_delta.ptr, i);
        const auto gamma_wing_local = gamma_wing.layout.subview(gamma_wing.ptr, i);
        const auto gamma_wing_prev_local = gamma_wing_prev.layout.subview(gamma_wing_prev.ptr, i);
        const auto velocities_local = velocities.layout.subview(velocities.ptr, i);
        auto forces_local = forces.layout.subview(forces.ptr, i);
        coeff_unsteady_cl_single_forces(
            verts_wing_local, 
            gamma_delta_local,
            gamma_wing_local,
            gamma_wing_prev_local,
            velocities_local,
            areas_local,
            normals[i],
            forces_local,
            freestream,
            dt
        );
    }
}

f32 Backend::coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& gamma_wake, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cd = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wake.layout.surfaces().size(); i++) {
        const auto verts_wake_local = verts_wake.layout.subview(verts_wake.ptr, i);
        const auto gamma_wake_local = gamma_wake.layout.subview(gamma_wake.ptr, i);
        const auto areas_local = areas.layout.subview(areas.ptr, i);
        const f32 area_local = mesh_area(areas_local);
        const f32 wing_cd = coeff_steady_cd_single(verts_wake_local, gamma_wake_local, flow, area_local);
        cd += wing_cd * area_local;
        total_area += area_local;
    }
    cd /= total_area;
    return cd;
}
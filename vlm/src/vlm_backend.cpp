#include "vlm_backend.hpp"
#include "vlm_utils.hpp"

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

f32 Backend::coeff_steady_cl_multi(const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_delta, const FlowData& flow, const MultiTensorView2D<Location::Device>& areas) {
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wing.size(); i++) {
        const f32 area_local = sum(areas[i]);
        const f32 wing_cl = coeff_steady_cl_single(
            verts_wing[i],
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

f32 Backend::coeff_steady_cd_multi(const MultiTensorView3D<Location::Device>& verts_wake, const MultiTensorView2D<Location::Device>& gamma_wake, const FlowData& flow, const MultiTensorView2D<Location::Device>& areas) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cd = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wake.size(); i++) {
        const f32 area_local = sum(areas[i]);
        const f32 wing_cd = coeff_steady_cd_single(
            verts_wake[i],
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

void Backend::wake_shed(const MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& verts_wake, i32 iteration) {
    for (const auto& [wing, wake] : zip(verts_wing, verts_wake)) {
        wing.slice(All, -1, All).to(wake.slice(All, -1-iteration, All));
    }
}

void Backend::displace_wing(const MultiTensorView2D<Location::Device>& transforms, MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& verts_wing_init) {
    // const tiny::ScopedTimer t("Mesh::move");

    // TODO: parallel for
    for (i64 i = 0; i < verts_wing.size(); i++) {
        const auto& verts_wing_i = verts_wing[i];
        const auto& verts_wing_init_i = verts_wing_init[i];
        const auto& transform_i = transforms[i];

        blas->gemm(
            1.0f,
            verts_wing_init_i.reshape(verts_wing_init_i.shape(0)*verts_wing_init_i.shape(1), 4),
            transform_i,
            0.0f,
            verts_wing_i.reshape(verts_wing_i.shape(0)*verts_wing_i.shape(1), 4),
            BLAS::Trans::No,
            BLAS::Trans::Yes
        );
    }
}

f32 Backend::coeff_cl_multibody(const MultiTensorView3D<Location::Device>& aero_forces, const MultiTensorView2D<Location::Device>& areas, const linalg::float3& freestream, f32 rho) {
    // parallel reduce
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 m = 0; m < aero_forces.size(); m++) {
        const f32 area_local = sum(areas[m]);
        const f32 wing_cl = coeff_cl(
            aero_forces[m],
            linalg::normalize(linalg::cross(freestream, {0.f, 1.f, 0.f})), // TODO: compute this from the wing frame
            freestream,
            rho,
            area_local
        );
        cl += wing_cl * area_local;
        total_area += area_local;
    }
    cl /= total_area;
    return cl;
}

void Backend::forces_unsteady_multibody(
    const MultiTensorView3D<Location::Device>& verts_wing,
    const MultiTensorView2D<Location::Device>& gamma_delta,
    const MultiTensorView2D<Location::Device>& gamma,
    const MultiTensorView2D<Location::Device>& gamma_prev,
    const MultiTensorView3D<Location::Device>& velocities,
    const MultiTensorView2D<Location::Device>& areas,
    const MultiTensorView3D<Location::Device>& normals,
    const MultiTensorView3D<Location::Device>& forces,
    f32 dt
) {
    // parallel tasks
    for (i64 m = 0; m < verts_wing.size(); m++) {
        forces_unsteady(
            verts_wing[m],
            gamma_delta[m],
            gamma[m],
            gamma_prev[m],
            velocities[m],
            areas[m],
            normals[m],
            forces[m],
            dt
        );
    }
}

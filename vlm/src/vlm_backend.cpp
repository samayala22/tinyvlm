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

// Anonymous namespace for internal helper templates
namespace {
    template<typename T>
    void wake_shed_impl(const MultiTensorView<T, 3, Location::Device>& verts_wing, MultiTensorView<T, 3, Location::Device>& verts_wake, i32 iteration) {
        for (const auto& [wing, wake] : zip(verts_wing, verts_wake)) {
            wing.slice(All, -1, All).to(wake.slice(All, -1-iteration, All));
        }
    }

    template<typename T>
    void displace_wing_impl(Backend* backend, const MultiTensorView<T, 2, Location::Device>& transforms, MultiTensorView<T, 3, Location::Device>& verts_wing, MultiTensorView<T, 3, Location::Device>& verts_wing_init) {
        // const tiny::ScopedTimer t("Mesh::move");

        // TODO: parallel for
        for (i64 i = 0; i < verts_wing.size(); i++) {
            const auto& verts_wing_i = verts_wing[i];
            const auto& verts_wing_init_i = verts_wing_init[i];
            const auto& transform_i = transforms[i];

            backend->blas->gemm(
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

    template<typename T>
    T coeff_cl_multibody_impl(
        Backend* backend,
        MultiTensorView<T, 3, Location::Device>& aero_forces,
        MultiTensorView<T, 2, Location::Device>& areas,
        const linalg::vec<T, 3>& freestream,
        T rho
    ) {
        // parallel reduce
        T cl = 0.0f;
        T total_area = 0.0f;
        for (i64 m = 0; m < aero_forces.size(); m++) {
            auto areas_m = areas[m];
            auto aero_m  = aero_forces[m];
            const T area_local = backend->sum(areas_m);
            auto lift_axis = linalg::normalize(linalg::cross(freestream, {(T)0.f, (T)1.f, (T)0.f}));
            auto fs = freestream;
            const T wing_cl = backend->coeff_cl(
                aero_m,
                lift_axis, // non-const lvalue
                fs,        // non-const lvalue
                rho,
                area_local
            );
            cl += wing_cl * area_local;
            total_area += area_local;
        }
        cl /= total_area;
        return cl;
    }

    template<typename T>
    linalg::vec<T, 3> coeff_cm_multibody_impl(
        Backend* backend,
        MultiTensorView<T, 3, Location::Device>& aero_forces,
        MultiTensorView<T, 3, Location::Device>& verts_wing,
        MultiTensorView<T, 2, Location::Device>& areas,
        const linalg::vec<T, 3>& ref_pt,
        const linalg::vec<T, 3>& freestream, 
        T rho
    )
    {
        linalg::vec<T, 3> cm = {(T)0.0f, (T)0.0f, (T)0.0f};
        T total_area = 0.0f;
        T total_mac = 0.0f;
        for (i64 m = 0; m < aero_forces.size(); m++) {
            auto areas_m = areas[m];
            auto verts_m = verts_wing[m];
            auto aero_m  = aero_forces[m];
            const T area_local = backend->sum(areas_m);
            const T mac_local = backend->mesh_mac(verts_m, areas_m);
            auto rp = ref_pt;
            auto fs = freestream;
            const auto local_cm = backend->coeff_cm(
                aero_m,
                verts_m,
                rp, // non-const lvalue
                fs, // non-const lvalue
                rho,
                area_local,
                mac_local
            );
            cm += local_cm * area_local * mac_local;
            total_area += area_local;
            total_mac += mac_local;
        }
        cm /= total_area * total_mac;
        return cm;
    }
} // namespace

// Remove const qualifier from verts_wing to match header
void Backend::wake_shed(MultiTensorView3fD& verts_wing, MultiTensorView3fD& verts_wake, i32 iteration) {
    wake_shed_impl(verts_wing, verts_wake, iteration);
}

void Backend::wake_shed(MultiTensorView3dD& verts_wing, MultiTensorView3dD& verts_wake, i32 iteration) {
    wake_shed_impl(verts_wing, verts_wake, iteration);
}

// Remove const qualifier from transforms to match header
void Backend::displace_wing(MultiTensorView2fD& transforms, MultiTensorView3fD& verts_wing, MultiTensorView3fD& verts_wing_init) {
    displace_wing_impl(this, transforms, verts_wing, verts_wing_init);
}

void Backend::displace_wing(MultiTensorView2dD& transforms, MultiTensorView3dD& verts_wing, MultiTensorView3dD& verts_wing_init) {
    displace_wing_impl(this, transforms, verts_wing, verts_wing_init);
}

// Remove const qualifiers from parameters to match header
f32 Backend::coeff_cl_multibody(
    MultiTensorView3fD& aero_forces,
    MultiTensorView2fD& areas,
    linalg::float3& freestream,
    f32 rho
)
{
    return coeff_cl_multibody_impl(this, aero_forces, areas, freestream, rho);
}

f64 Backend::coeff_cl_multibody(
    MultiTensorView3dD& aero_forces,
    MultiTensorView2dD& areas,
    linalg::double3& freestream,
    f64 rho
)
{
    return coeff_cl_multibody_impl(this, aero_forces, areas, freestream, rho);
}

// Remove const qualifiers from parameters to match header
linalg::float3 Backend::coeff_cm_multibody(
    MultiTensorView3fD& aero_forces,
    MultiTensorView3fD& verts_wing,
    MultiTensorView2fD& areas,
    linalg::float3& ref_pt,
    linalg::float3& freestream, 
    f32 rho
) {
    return coeff_cm_multibody_impl(this, aero_forces, verts_wing, areas, ref_pt, freestream, rho);
}

linalg::double3 Backend::coeff_cm_multibody(
    MultiTensorView3dD& aero_forces,
    MultiTensorView3dD& verts_wing,
    MultiTensorView2dD& areas,
    linalg::double3& ref_pt,
    linalg::double3& freestream, 
    f64 rho
) {
    return coeff_cm_multibody_impl(this, aero_forces, verts_wing, areas, ref_pt, freestream, rho);
}

// Remove const qualifiers from parameters to match header
void Backend::forces_unsteady_multibody(
    MultiTensorView3fD& verts_wing,
    MultiTensorView2fD& gamma_delta,
    MultiTensorView2fD& dgamma_dt,
    MultiTensorView3fD& velocities,
    MultiTensorView2fD& areas,
    MultiTensorView3fD& normals,
    MultiTensorView3fD& forces
) {
    // parallel tasks
    for (i64 m = 0; m < verts_wing.size(); m++) {
        forces_unsteady(
            verts_wing[m],
            gamma_delta[m],
            dgamma_dt[m],
            velocities[m],
            areas[m],
            normals[m],
            forces[m]
        );
    }
}

void Backend::forces_unsteady_multibody(
    MultiTensorView3dD& verts_wing,
    MultiTensorView2dD& gamma_delta,
    MultiTensorView2dD& dgamma_dt,
    MultiTensorView3dD& velocities,
    MultiTensorView2dD& areas,
    MultiTensorView3dD& normals,
    MultiTensorView3dD& forces
) {
    // parallel tasks
    for (i64 m = 0; m < verts_wing.size(); m++) {
        forces_unsteady(
            verts_wing[m],
            gamma_delta[m],
            dgamma_dt[m],
            velocities[m],
            areas[m],
            normals[m],
            forces[m]
        );
    }
}

// Remove const qualifiers from parameters to match header
void Backend::forces_steady_multibody(
    MultiTensorView3fD& verts_wing,
    MultiTensorView2fD& gamma_delta,
    MultiTensorView3fD& velocities,
    MultiTensorView3fD& forces
) {
    for (i64 m = 0; m < verts_wing.size(); m++) {
        forces_steady(
            verts_wing[m],
            gamma_delta[m],
            velocities[m],
            forces[m]
        );
    }
}

void Backend::forces_steady_multibody(
    MultiTensorView3dD& verts_wing,
    MultiTensorView2dD& gamma_delta,
    MultiTensorView3dD& velocities,
    MultiTensorView3dD& forces
) {
    for (i64 m = 0; m < verts_wing.size(); m++) {
        forces_steady(
            verts_wing[m],
            gamma_delta[m],
            velocities[m],
            forces[m]
        );
    }
}
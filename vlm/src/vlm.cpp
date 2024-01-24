#include "vlm.hpp"
// #include "vlm_solver.hpp"
#include "vlm_backend.hpp"
#include "tinycpuid.hpp"
#include "tinyinterpolate.hpp"

#ifdef VLM_AVX2
#include "vlm_backend_avx2.hpp"
#endif
#ifdef VLM_CUDA
#include "vlm_backend_cuda.hpp"
#endif

#include <iostream>
#include <cstdio>
#include <memory>

using namespace vlm;

VLM::VLM(tiny::Config& cfg) : mesh(cfg), data(cfg) {};

void VLM::init() {
    mesh.compute_connectivity();
    mesh.compute_metrics_wing();
    if (data.c_ref == 0.0f) data.c_ref = mesh.chord_mean(0, mesh.ns+1);
    if (data.s_ref == 0.0f) data.s_ref = mesh.panels_area_xy(0,0, mesh.nc, mesh.ns);
    data.alloc(mesh.nc*mesh.ns);
}

// Backend factory
// Note: should refactor this to separate CPU and GPU backends
std::unique_ptr<Backend> create_backend(const std::string& backend_name, Mesh& mesh, Data& data) {
    std::unique_ptr<Backend> backend;
    tiny::CPUID cpuid;
    cpuid.print_info();

    #ifdef VLM_AVX2
    if (backend_name == "avx2" && cpuid.has("AVX2")) {
        backend = std::make_unique<BackendAVX2>(mesh, data);
        return backend;
    }
    #endif
    #ifdef VLM_CUDA
    if (backend_name == "cuda") {
        backend = std::make_unique<BackendCUDA>(mesh, data);
        return backend;
    }
    #endif
    throw std::runtime_error("Unsupported backend: " + backend_name);
}

f32 to_radians(f32 degrees) {
    return degrees * PI_f / 180.0f;
}

void VLM::solve(tiny::Config& cfg) {
    std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");

    auto backend = create_backend(backend_name, mesh, data);

    std::vector<f32> alphas = cfg().section("solver").get_vector<f32>("alphas", {0.0f});
    for (auto alpha : alphas) {
        const f32 alpha_rad = to_radians(alpha);
        backend->reset();
        data.reset();
        data.compute_freestream(alpha_rad);
        mesh.update_wake(data.u_inf);
        mesh.correction_high_aoa(alpha_rad); // must be after update_wake
        backend->compute_lhs();
        backend->compute_rhs();
        backend->lu_factor();
        backend->lu_solve();
        backend->compute_delta_gamma();
        data.cl = backend->compute_coefficient_cl(mesh, data, data.s_ref);
        data.cd = backend->compute_coefficient_cd(mesh, data, data.s_ref);
        data.cm = backend->compute_coefficient_cm(mesh, data, data.s_ref, data.c_ref);
        std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm.x(), data.cm.y(), data.cm.z());
    }
}

void VLM::solve_nonlinear(tiny::Config& cfg) {
    std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");
    std::string file_database = cfg().section("files").get<std::string>("database");
    std::vector<f32> alphas = cfg().section("solver").get_vector<f32>("alphas", {0.0f});

    std::vector<f32> strip_alphas(mesh.ns);
    std::vector<f32> db_alphas;
    std::vector<f32> db_cl_visc;
    // read db file and set those two vectors
    tiny::AkimaInterpolator<f32> interpolator{};
    interpolator.set_data(db_alphas, db_cl_visc);

    auto backend = create_backend(backend_name, mesh, data);

    const f32 van_dam_tol = 1e-6f;
    f32 van_dam_err = 1.0f; // l1 error
    const u32 max_iter = 100;

    for (auto alpha : alphas) {
        const f32 alpha_rad = to_radians(alpha);

        backend->reset();
        data.reset();
        data.compute_freestream(alpha_rad);
        mesh.update_wake(data.u_inf);
        mesh.correction_high_aoa(alpha_rad); // must be after update_wake
        backend->compute_lhs();
        backend->lu_factor();
        // backend->compute_rhs();
        // backend->lu_solve();
        // backend->compute_delta_gamma();
        // data.cl = backend->compute_coefficient_cl(mesh, data, data.s_ref);
        // data.cd = backend->compute_coefficient_cd(mesh, data, data.s_ref);
        // data.cm = backend->compute_coefficient_cm(mesh, data, data.s_ref, data.c_ref);
        // std::printf(">>> Before Coupling : Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm.x(), data.cm.y(), data.cm.z());

        for (u32 iter = 0; iter < max_iter && van_dam_err > van_dam_tol; iter++) {
            van_dam_err = 0.0f; // reset l1 error
            backend->rebuild_rhs(strip_alphas);
            backend->lu_solve();
            backend->compute_delta_gamma();
            
            // parallel reduce
            for (u32 j = 0; j < mesh.ns; j++) {
                const f32 strip_area = mesh.panels_area(0, j, mesh.nc, 1);
                const f32 strip_cl = backend->compute_coefficient_cl(mesh, data, strip_area, j, 1);
                const f32 effective_aoa = strip_cl / (2*PI_f) - strip_alphas[j] + alpha_rad;
                const f32 correction = (interpolator(effective_aoa) - strip_cl) / (2*PI_f);
                strip_alphas[j] += correction;
                van_dam_err += std::abs(correction);
            }
            van_dam_err /= mesh.ns; // normalize l1 error
            std::printf(">>> Iter: %d | Error: %.3e \n", iter, van_dam_err);
        }

        data.cl = backend->compute_coefficient_cl(mesh, data, data.s_ref);
        data.cd = backend->compute_coefficient_cd(mesh, data, data.s_ref);
        data.cm = backend->compute_coefficient_cm(mesh, data, data.s_ref, data.c_ref);
        std::printf(">>> After Coupling : Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm.x(), data.cm.y(), data.cm.z());
    }
}
#include "vlm.hpp"
// #include "vlm_solver.hpp"
#include "vlm_backend.hpp"
#include "tinycpuid.hpp"

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
        data.cl = backend->compute_coefficient_cl(mesh, data, 0, mesh.ns, data.s_ref);
        data.cd = backend->compute_coefficient_cd(mesh, data, 0, mesh.ns, data.s_ref);
        data.cm = backend->compute_coefficient_cm(mesh, data, 0, mesh.ns, data.s_ref, data.c_ref);
        std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm.x(), data.cm.y(), data.cm.z());
    }

    // std::cout << mesh.chord_mean(0, mesh.ns+1) << std::endl;
    // std::cout << mesh.panels_area_xy(0,0, mesh.nc, mesh.ns) << std::endl;
    // std::cout << mesh.panels_area(0,0, mesh.nc, mesh.ns) << std::endl;

    // Pause for memory reading
    // std::cout << "Done ..." << std::endl;
    // std::cin.get();
}

// void VLM::solve_nonlinear(tiny::Config& cfg) {
//     std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");

//     auto backend = create_backend(backend_name, mesh, data);

//     std::vector<f32> alphas = cfg().section("solver").get_vector<f32>("alphas", {0.0f});
//     for (auto alpha : alphas) {
//         const f32 alpha_rad = to_radians(alpha);
//         backend->reset();
//         data.reset();
//         data.compute_freestream(alpha_rad);
//         mesh.update_wake(data.u_inf);
//         mesh.correction_high_aoa(alpha_rad); // must be after update_wake
//         backend->compute_lhs();
//         backend->compute_rhs();
//         backend->lu_factor();
//         backend->lu_solve();
//         backend->compute_delta_gamma();
//         backend->compute_coefficients();
//         std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm.x(), data.cm.y(), data.cm.z());
//     }

//     // std::cout << mesh.chord_mean(0, mesh.ns+1) << std::endl;
//     // std::cout << mesh.panels_area_xy(0,0, mesh.nc, mesh.ns) << std::endl;
//     // std::cout << mesh.panels_area(0,0, mesh.nc, mesh.ns) << std::endl;

//     // Pause for memory reading
//     // std::cout << "Done ..." << std::endl;
//     // std::cin.get();
// }
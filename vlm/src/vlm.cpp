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
    data.alloc(mesh.nc*mesh.ns);
}

// Backend factory
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

void VLM::solve(tiny::Config& cfg) {
    std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");

    auto backend = create_backend(backend_name, mesh, data);

    std::vector<f32> alphas = cfg().section("solver").get_vector<f32>("alphas", {0.0f});
    for (auto alpha : alphas) {
        const f32 alpha_rad = alpha * PI_f / 180.0f;
        backend->reset();
        data.reset();
        data.compute_freestream(alpha_rad);
        mesh.update_wake(data.u_inf);
        mesh.correction_high_aoa(alpha_rad); // must be after update_wake
        backend->compute_lhs();
        backend->compute_rhs();
        backend->solve();
        backend->compute_delta_gamma();
        backend->compute_forces();
        std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm_x, data.cm_y, data.cm_z);
    }

    // Pause for memory reading
    // std::cout << "Done ..." << std::endl;
    // std::cin.get();
}
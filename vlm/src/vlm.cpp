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
        backend->reset();
        data.reset();
        data.alpha = to_radians(alpha);
        data.beta = to_radians(0.0f);
        mesh.update_wake(data.freestream());
        mesh.correction_high_aoa(data.alpha); // must be after update_wake
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

struct SpallartLiftCurve {
    const f32 cl_0, a0, a1, cl_a0, cl_a1;
    SpallartLiftCurve(f32 cl_0_, f32 a0_, f32 a1_, f32 cl_a0_, f32 cl_a1_):
        cl_0(cl_0_), a0(a0_), a1(a1_), cl_a0(cl_a0_), cl_a1(cl_a1_) {}
    inline f32 operator()(f32 alpha) const {
        return cl_a0 * alpha + 0.5f * (cl_0 - cl_a1 * alpha) * (1.f + std::erf((alpha - a0) / a1));
    }
};

template<typename T>
void linspace(T start, T end, u32 n, std::vector<T>& out) {
    out.resize(n);
    T step = (end - start) / (n - 1);
    for (u32 i = 0; i < n; i++) {
        out[i] = start + i * step;
    }
}

void VLM::solve_nonlinear(tiny::Config& cfg) {
    std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");
    std::string file_database = cfg().section("files").get<std::string>("database");
    //std::vector<f32> alphas = cfg().section("solver").get_vector<f32>("alphas", {0.0f});
    
    // Temporary setup for testing
    // TODO: once it works, move this into a testing framework
    // std::vector<f32> alphas = {0, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<f32> alphas = {5};
    std::vector<f32> strip_alphas(mesh.ns);

    std::vector<f32> db_alphas;
    linspace(0.f, 20.f, 100, db_alphas);

    std::vector<f32> db_cl_visc(db_alphas.size());
    SpallartLiftCurve lift_curve(1.2f, 0.28f, 0.02f, 2.f*PI_f, 2.f*PI_f);

    for (int i = 0; i < db_alphas.size(); i++) {
        db_cl_visc[i] = lift_curve(db_alphas[i]);
    }
    tiny::AkimaInterpolator<f32> interpolator{};
    interpolator.set_data(db_alphas, db_cl_visc);

    auto backend = create_backend(backend_name, mesh, data);

    const f32 van_dam_tol = 1e-4f;
    f32 van_dam_err = 1.0f; // l1 error
    const u32 max_iter = 100;

    for (auto alpha : alphas) {
        backend->reset();
        data.reset();
        data.alpha = to_radians(alpha);
        data.beta = to_radians(0.0f);
        mesh.update_wake(data.freestream());
        mesh.correction_high_aoa(data.alpha); // must be after update_wake
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
            backend->compute_rhs(strip_alphas);
            backend->lu_solve();
            backend->compute_delta_gamma();
            
            // parallel reduce
            for (u32 j = 0; j < mesh.ns; j++) {
                const f32 strip_area = mesh.panels_area(0, j, mesh.nc, 1);
                const f32 strip_alpha = strip_alphas[j];
                const auto strip_freestream = data.freestream(strip_alpha, 0.0f);
                const f32 strip_cl = backend->compute_coefficient_cl(mesh, data, strip_area, strip_freestream, j, 1);
                const f32 effective_aoa = strip_cl / (2*PI_f) - strip_alphas[j] + data.alpha;
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
        std::printf(">>> Analytical CL: %.6f\n", lift_curve(data.alpha));
    }
}
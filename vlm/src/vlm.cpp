#include "vlm.hpp"

#include "vlm_backend.hpp"
#include "tinycpuid.hpp"
#include "tinyinterpolate.hpp"
#include <limits>
#include <utility>

#ifdef VLM_AVX2
#include "vlm_backend_avx2.hpp"
#endif
#ifdef VLM_CUDA
#include "vlm_backend_cuda.hpp"
#endif

#include <iostream>
#include <cstdio>
#include <memory>
#include <algorithm>

using namespace vlm;

// Backend factory
// Note: should refactor this to separate CPU and GPU backends
inline std::unique_ptr<Backend> create_backend(const std::string& backend_name, Mesh& mesh) {
    std::unique_ptr<Backend> backend;
    tiny::CPUID cpuid;
    cpuid.print_info();

    #ifdef VLM_AVX2
    if (backend_name == "avx2" && cpuid.has("AVX2")) {
        backend = std::make_unique<BackendAVX2>(mesh);
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

Solver::Solver(const tiny::Config& cfg) : mesh(cfg) {
    std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");
    backend = create_backend(backend_name, mesh);
}

void LinearVLM::solve(const FlowData& flow) {
    backend->reset();
    mesh.update_wake(flow.freestream);
    mesh.correction_high_aoa(flow.alpha); // must be after update_wake
    backend->compute_lhs(flow);
    backend->compute_rhs(flow);
    backend->lu_factor();
    backend->lu_solve();
    backend->compute_delta_gamma();
    auto cl = backend->compute_coefficient_cl(flow);
    auto cd = backend->compute_coefficient_cd(flow);
    auto cm = backend->compute_coefficient_cm(flow);
    std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", flow.alpha, cl, cd, cm.x(), cm.y(), cm.z());
}

// void NonLinearVLM::solve() {
//     std::printf("-------------- Alpha = %.1f ---------------\n", data.alpha);
//     f64 err = 1.0f; // l1 error
//     strip_alphas.resize(mesh.ns);
//     std::fill(strip_alphas.begin(), strip_alphas.end(), data.alpha); // memset

//     backend->reset();
//     mesh.update_wake(data.freestream()); // Create wake panels in freestream axis
//     mesh.correction_high_aoa(data.alpha); // Correct collocation point
//     backend->compute_lhs(); // Create influence matrix
//     backend->lu_factor(); // Factorize the influence matrix into LU form

//     for (u32 iter = 0; iter < max_iter && err > tol; iter++) {
//         err = 0.0; // reset l1 error
//         backend->compute_rhs(strip_alphas); // Compute RHS using strip alphas
//         backend->lu_solve(); // Solve for the gammas
//         backend->compute_delta_gamma(); // Compute the chordwise delta gammas for force computation
        
//         // parallel reduce
//         // loop over the chordwise strips and apply Van Dam algorithm
//         for (u32 j = 0; j < mesh.ns; j++) {
//             const f32 strip_area = mesh.panels_area(0, j, mesh.nc, 1);
//             const f32 strip_alpha = strip_alphas[j];
//             const auto strip_freestream = data.freestream(strip_alpha, 0.0f);
//             const f32 strip_cl = backend->compute_coefficient_cl(mesh, data, strip_area, strip_freestream, j, 1);
//             const f32 effective_aoa = strip_cl / (2.f*PI_f) - strip_alpha + data.alpha;
//             // TODO: interpolated value should be computed at the y mid point of the strip
//             const f32 correction = (database.interpolate_CL(effective_aoa, 0.f) - strip_cl) / (2.f*PI_f);
//             strip_alphas[j] += correction;
//             err += (f64)std::abs(correction);
//         }
//         err /= (f64)mesh.ns; // normalize l1 error
//         // std::printf(">>> Iter: %d | Error: %.3e \n", iter, err);
//     }

//     data.cl = backend->compute_coefficient_cl(mesh, data, data.s_ref);
//     data.cd = backend->compute_coefficient_cd(mesh, data, data.s_ref);
//     data.cm = backend->compute_coefficient_cm(mesh, data, data.s_ref, data.c_ref);
// }

// void VLM::solve_nonlinear(tiny::Config& cfg) {
//     std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");
//     std::string file_database = cfg().section("files").get<std::string>("database");
//     //std::vector<f32> alphas = cfg().section("solver").get_vector<f32>("alphas", {0.0f});
    
//     // Temporary setup for testing
//     // TODO: once it works, move this into a testing framework
//     // std::vector<f32> alphas = {0, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
//     std::vector<f32> alphas = {20};
//     std::vector<f32> strip_alphas(mesh.ns);

//     std::vector<f32> db_alphas;
//     linspace(to_radians(0.f), to_radians(20.f), 100, db_alphas);

//     std::vector<f32> db_cl_visc(db_alphas.size());
//     //SpallartLiftCurve lift_curve(1.2f, 0.28f, 0.02f, 2.f*PI_f, 2.f*PI_f);
//     ThinAirfoilPolarLiftCurve lift_curve{};

//     for (int i = 0; i < db_alphas.size(); i++) {
//         db_cl_visc[i] = lift_curve(db_alphas[i]);
//     }
//     //tiny::AkimaInterpolator<f32> interpolator{db_alphas, db_cl_visc};
//     tiny::LinearInterpolator<f32> interpolator{db_alphas, db_cl_visc};

//     auto backend = create_backend(backend_name, mesh, data);

//     for (auto alpha : alphas) {
//         std::printf("-------------- Alpha = %.1f ---------------\n", alpha);
//         const f64 van_dam_tol = 1e-4f;
//         f64 van_dam_err = 1.0f; // l1 error
//         const u32 max_iter = 100;

//         backend->reset();
//         data.reset();
//         data.alpha = to_radians(alpha);
//         data.beta = to_radians(0.0f);
//         mesh.update_wake(data.freestream()); // Create wake panels in freestream axis
//         mesh.correction_high_aoa(data.alpha); // Correct collocation point
//         backend->compute_lhs(); // Create influence matrix
//         backend->lu_factor(); // Factorize the influence matrix into LU form
//         std::fill(strip_alphas.begin(), strip_alphas.end(), data.alpha);
        
//         backend->compute_rhs();
//         backend->lu_solve();
//         backend->compute_delta_gamma();
//         data.cl = backend->compute_coefficient_cl(mesh, data, data.s_ref);
//         data.cd = backend->compute_coefficient_cd(mesh, data, data.s_ref);
//         data.cm = backend->compute_coefficient_cm(mesh, data, data.s_ref, data.c_ref);
//         std::printf(">>> Before Coupling : Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm.x(), data.cm.y(), data.cm.z());

//         for (u32 iter = 0; iter < max_iter && van_dam_err > van_dam_tol; iter++) {
//             van_dam_err = 0.0; // reset l1 error
//             backend->compute_rhs(strip_alphas); // Compute RHS using strip alphas
//             backend->lu_solve(); // Solve for the gammas
//             backend->compute_delta_gamma(); // Compute the chordwise delta gammas for force computation
            
//             // parallel reduce
//             // loop over the chordwise strips and apply Van Dam algorithm
//             for (u32 j = 0; j < mesh.ns; j++) {
//                 const f32 strip_area = mesh.panels_area(0, j, mesh.nc, 1);
//                 const f32 strip_alpha = strip_alphas[j];
//                 const auto strip_freestream = data.freestream(strip_alpha, 0.0f);
//                 const f32 strip_cl = backend->compute_coefficient_cl(mesh, data, strip_area, strip_freestream, j, 1);
//                 const f32 effective_aoa = strip_cl / (2.f*PI_f) - strip_alpha + data.alpha;
//                 const f32 correction = (interpolator(effective_aoa) - strip_cl) / (2.f*PI_f);
//                 // if (j == 0) {
//                 //     std::printf("strip cl: %.6f | effective aoa: %.6f | aoa: %.6f \n", strip_cl, effective_aoa, data.alpha);
//                 //     std::printf("correction: %.6f \n", correction);
//                 // }
//                 strip_alphas[j] += correction;
//                 van_dam_err += (f64)std::abs(correction);
//             }
//             van_dam_err /= (f64)mesh.ns; // normalize l1 error
//             std::printf(">>> Iter: %d | Error: %.3e \n", iter, van_dam_err);
//         }

//         data.cl = backend->compute_coefficient_cl(mesh, data, data.s_ref);
//         data.cd = backend->compute_coefficient_cd(mesh, data, data.s_ref);
//         data.cm = backend->compute_coefficient_cm(mesh, data, data.s_ref, data.c_ref);
//         std::printf(">>> After Coupling : Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", alpha, data.cl, data.cd, data.cm.x(), data.cm.y(), data.cm.z());
//         const f32 abs_error = std::abs(data.cl - lift_curve(data.alpha));
//         std::printf(">>> Analytical: %.6f | Abs Error: %.3E | Relative Error: %.5f%% \n", lift_curve(data.alpha), abs_error, 100.0f * abs_error / (lift_curve(data.alpha) + std::numeric_limits<f32>::epsilon()));
//     }
// }
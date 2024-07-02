#include "vlm.hpp"

#include "vlm_backend.hpp"
#include "tinytimer.hpp"
#include "tinyconfig.hpp"

#include "vlm_data.hpp"
#include "vlm_mesh.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"

#include <iostream>
#include <cstdio>
#include <memory>
#include <algorithm>

using namespace vlm;

// Solver::Solver(const tiny::Config& cfg) {
//     std::string backend_name = cfg().section("solver").get<std::string>("backend", "cpu");
//     mesh = std::make_unique<Mesh>(cfg);
//     backend = create_backend(backend_name, *mesh);
// }

AeroCoefficients LinearVLM::solve(const FlowData& flow) {
    auto init_pos = translation_matrix<f32>({
        -100.0f * flow.u_inf*std::cos(flow.alpha),
        0.0f,
        -100.0f * flow.u_inf*std::sin(flow.alpha)
    });

    backend->reset();
    //backend->update_wake(flow.freestream);
    backend->mesh_move(init_pos);
    backend->mesh_metrics(flow.alpha);
    backend->lhs_assemble();
    backend->set_velocities(flow.freestream);
    backend->compute_rhs();
    backend->lu_factor();
    backend->lu_solve();
    backend->compute_delta_gamma();
    return AeroCoefficients{
        backend->compute_coefficient_cl(flow),
        backend->compute_coefficient_cd(flow),
        backend->compute_coefficient_cm(flow)
    };
}

void strip_alpha_to_vel(const FlowData& flow, Mesh2* mesh, f32* local_velocities, f32* strip_alphas) {
    const u64 nb_panels_wing = mesh_nb_panels_wing(mesh);
    f32* lvx = local_velocities + 0 * nb_panels_wing;
    f32* lvy = local_velocities + 1 * nb_panels_wing;
    f32* lvz = local_velocities + 2 * nb_panels_wing;
    for (u64 j = 0; j < mesh->ns; j++) {
        auto fs = compute_freestream(flow.u_inf, strip_alphas[j], 0.0f);
        lvx[j] = fs.x;
        lvy[j] = fs.y;
        lvz[j] = fs.z;
    }
    for (u64 i = 1; i < mesh->nc; i++) {
        std::memcpy(lvx + i * mesh->ns, lvx, mesh->ns * sizeof(*lvx));
        std::memcpy(lvy + i * mesh->ns, lvy, mesh->ns * sizeof(*lvy));
        std::memcpy(lvz + i * mesh->ns, lvz, mesh->ns * sizeof(*lvz));
    }
}

AeroCoefficients NonLinearVLM::solve(const FlowData& flow, const Database& db) {
    f64 err = 1.0f; // l1 error
    auto init_pos = translation_matrix<f32>({
        -100.0f * flow.u_inf*std::cos(flow.alpha),
        0.0f,
        -100.0f * flow.u_inf*std::sin(flow.alpha)
    });

    std::fill(strip_alphas, strip_alphas + backend->hd_mesh->ns, flow.alpha); // memset

    backend->reset();
    backend->mesh_move(init_pos);
    backend->mesh_metrics(flow.alpha);
    backend->lhs_assemble();
    backend->lu_factor();

    for (u64 iter = 0; iter < max_iter && err > tol; iter++) {
        err = 0.0; // reset l1 error
        strip_alpha_to_vel(flow, backend->hd_mesh, velocities, strip_alphas); // Compute local panel velocities based on strip alphas
        backend->allocator.hd_memcpy(backend->dd_data->local_velocities, velocities, 3 * mesh_nb_panels_wing(backend->hd_mesh) * sizeof(*velocities));
        backend->compute_rhs(); // Compute RHS using strip alphas (on CPU)
        backend->lu_solve(); // Copy RHS on device and solve for gamma
        backend->compute_delta_gamma(); // Compute the chordwise delta gammas for force computation
        
        // Parallel Reduce
        // loop over the chordwise strips and apply Van Dam algorithm
        for (u64 j = 0; j < backend->hd_mesh->ns; j++) {
            const f32 strip_area = backend->mesh_area(0, j, backend->hd_mesh->nc, 1);
            const FlowData strip_flow = {strip_alphas[j], flow.beta, flow.u_inf, flow.rho};
            const f32 strip_cl = backend->compute_coefficient_cl(strip_flow, strip_area, j, 1);
            const f32 effective_aoa = strip_cl / (2.f*PI_f) - strip_flow.alpha + flow.alpha;

            // TODO: interpolated value should be computed at the y mid point of the strip
            const f32 correction = (db.interpolate_CL(effective_aoa, 0.f) - strip_cl) / (2.f*PI_f);
            // std::printf(">>> Strip: %d | CL: %.3f | Interpolated: %.3f | Correction: %.3e\n", j, strip_cl, db.interpolate_CL(effective_aoa, 0.f), correction);
            strip_alphas[j] += correction;
            err += (f64)std::abs(correction);
        }
        err /= (f64)backend->hd_mesh->ns; // normalize l1 error
        //std::printf(">>> Iter: %d | Error: %.3e \n", iter, err);
    }
    return AeroCoefficients{
        backend->compute_coefficient_cl(flow),
        backend->compute_coefficient_cd(flow),
        backend->compute_coefficient_cm(flow)
    };
}
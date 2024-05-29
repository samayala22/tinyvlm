#include "vlm.hpp"

#include "vlm_backend.hpp"
#include "tinytimer.hpp"
#include "tinyconfig.hpp"

#include "vlm_data.hpp"
#include "vlm_mesh.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include <utility>

#include <iostream>
#include <cstdio>
#include <memory>
#include <algorithm>

using namespace vlm;

Solver::Solver(const tiny::Config& cfg) {
    std::string backend_name = cfg().section("solver").get<std::string>("backend", "cpu");
    mesh = std::make_unique<Mesh>(cfg);
    backend = create_backend(backend_name, *mesh);
}

AeroCoefficients LinearVLM::solve(const FlowData& flow) {
    backend->reset();
    backend->set_velocities(flow.freestream);
    mesh->update_wake(flow.freestream);
    mesh->correction_high_aoa(flow.alpha); // must be after update_wake
    backend->lhs_assemble();
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

AeroCoefficients NonLinearVLM::solve(const FlowData& flow, const Database& db) {
    f64 err = 1.0f; // l1 error
    strip_alphas.resize(mesh->ns);
    SoA_3D_t<f32> velocities;
    velocities.resize(mesh->nb_panels_wing());

    std::fill(strip_alphas.begin(), strip_alphas.end(), flow.alpha); // memset

    backend->reset();
    mesh->update_wake(flow.freestream); // Create wake panels in freestream axis
    mesh->correction_high_aoa(flow.alpha); // Correct collocation point
    backend->lhs_assemble(); // Create influence matrix (copy mesh to device)
    backend->lu_factor(); // LU factorization on device

    for (u64 iter = 0; iter < max_iter && err > tol; iter++) {
        err = 0.0; // reset l1 error
        // BEGIN TODO: cleanup this
        for (u64 j = 0; j < mesh->ns; j++) {
            auto fs = compute_freestream(1.0f, strip_alphas[j], 0.0f);
            velocities.x[j] = fs.x;
            velocities.y[j] = fs.y;
            velocities.z[j] = fs.z;
        }
        for (u64 i = 1; i < mesh->nc; i++) {
            std::copy(velocities.x.data(), velocities.x.data()+mesh->ns, velocities.x.data() + i*mesh->ns);
            std::copy(velocities.y.data(), velocities.y.data()+mesh->ns, velocities.y.data() + i*mesh->ns);
            std::copy(velocities.z.data(), velocities.z.data()+mesh->ns, velocities.z.data() + i*mesh->ns);
        }
        // END TODO

        backend->set_velocities(velocities); // Set local velocities at collocation points
        backend->compute_rhs(); // Compute RHS using strip alphas (on CPU)
        backend->lu_solve(); // Copy RHS on device and solve for gamma
        backend->compute_delta_gamma(); // Compute the chordwise delta gammas for force computation
        
        // parallel reduce
        // loop over the chordwise strips and apply Van Dam algorithm
        {
            const tiny::ScopedTimer timer("Strip correction");
            for (u64 j = 0; j < mesh->ns; j++) {
                const f32 strip_area = mesh->panels_area(0, j, mesh->nc, 1);
                const FlowData strip_flow = {strip_alphas[j], flow.beta, flow.u_inf, flow.rho};
                const f32 strip_cl = backend->compute_coefficient_cl(strip_flow, strip_area, j, 1);
                const f32 effective_aoa = strip_cl / (2.f*PI_f) - strip_flow.alpha + flow.alpha;

                // TODO: interpolated value should be computed at the y mid point of the strip
                const f32 correction = (db.interpolate_CL(effective_aoa, 0.f) - strip_cl) / (2.f*PI_f);
                // std::printf(">>> Strip: %d | CL: %.3f | Interpolated: %.3f | Correction: %.3e\n", j, strip_cl, db.interpolate_CL(effective_aoa, 0.f), correction);
                strip_alphas[j] += correction;
                err += (f64)std::abs(correction);
            }
        }
        err /= (f64)mesh->ns; // normalize l1 error
        //std::printf(">>> Iter: %d | Error: %.3e \n", iter, err);
    }
    return AeroCoefficients{
        backend->compute_coefficient_cl(flow),
        backend->compute_coefficient_cd(flow),
        backend->compute_coefficient_cm(flow)
    };
}
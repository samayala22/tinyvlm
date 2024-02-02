#include "vlm.hpp"

#include "vlm_backend.hpp"
#include "tinytimer.hpp"

#include "vlm_data.hpp"
#include <utility>

#include <iostream>
#include <cstdio>
#include <memory>
#include <algorithm>

using namespace vlm;

Solver::Solver(const tiny::Config& cfg) {
    std::string backend_name = cfg().section("solver").get<std::string>("backend", "avx2");
    mesh = std::make_unique<Mesh>(cfg);
    backend = create_backend(backend_name, *mesh);
}

AeroCoefficients LinearVLM::solve(const FlowData& flow) {
    backend->reset();
    mesh->update_wake(flow.freestream);
    mesh->correction_high_aoa(flow.alpha); // must be after update_wake
    backend->compute_lhs(flow);
    backend->compute_rhs(flow);
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
    std::fill(strip_alphas.begin(), strip_alphas.end(), flow.alpha); // memset

    backend->reset();
    mesh->update_wake(flow.freestream); // Create wake panels in freestream axis
    mesh->correction_high_aoa(flow.alpha); // Correct collocation point
    backend->compute_lhs(flow); // Create influence matrix
    backend->lu_factor(); // Factorize the influence matrix into LU form

    for (u32 iter = 0; iter < max_iter && err > tol; iter++) {
        err = 0.0; // reset l1 error
        backend->compute_rhs(flow, strip_alphas); // Compute RHS using strip alphas
        backend->lu_solve(); // Solve for the gammas
        backend->compute_delta_gamma(); // Compute the chordwise delta gammas for force computation
        
        // parallel reduce
        // loop over the chordwise strips and apply Van Dam algorithm
        {
            const tiny::Timer timer("Strip correction");
            for (u32 j = 0; j < mesh->ns; j++) {
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
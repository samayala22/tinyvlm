#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>

#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_utils.hpp"

using namespace vlm;
using namespace linalg::ostream_overloads;

int main() {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_5x200.x"};
    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    // Define simulation length
    const f32 t_final = 10.0f;

    // Define wing motion
    auto displacement_wing = [&](f32 t) -> linalg::alias::float4x4 {
        return linalg::translation_matrix(linalg::alias::float3{
            0.0f,
            0.0f,
            std::sin(0.2f * t)
        });
    };

    // For the moment take AoA=0 and U_inf=1
    auto displacement_freestream = [&](f32 t) -> linalg::alias::float4x4 {
        return linalg::translation_matrix(linalg::alias::float3{
            -1.0f*t,
            0.0f,
            0.0f
        });
    };

    auto displacement = [&](f32 t) -> linalg::alias::float4x4 {
        return linalg::mul(displacement_freestream(t), displacement_wing(t));
    };

    auto relative_displacement = [&](f32 t0, f32 t1) -> linalg::alias::float4x4 {
        return linalg::mul(displacement(t1), linalg::inverse(displacement(t0)));
    };

    // Magnitude of the velocity of a vertex at time t calculated by finite difference
    auto absolute_velocity = [&](f32 t, const linalg::alias::float4& vertex) -> f32 {
        return linalg::length((linalg::mul(displacement(t+EPS_sqrt_f), vertex)-linalg::mul(displacement(t), vertex))/EPS_sqrt_f);
    };

    for (const auto& [mesh_name, backend_name] : solvers) {
        UVLM solver{};
        solver.mesh = create_mesh(mesh_name);
        solver.backend = create_backend(backend_name, *solver.mesh);

        std::vector<f32> vec_t; // pre-calculated timesteps

        // Pre-calculate timesteps to determine wake size
        vec_t.push_back(0.0f);
        for (f32 t = 0.0f; t < t_final;) {
            const f32 dt = (0.5f * (solver.mesh->chord_length(0) + solver.mesh->chord_length(1))) / absolute_velocity(t, {0.f, 0.f, 0.f, 1.f});
            t += dt;
            vec_t.push_back(t);
        }

        // TODO: think about splitting mesh body and wake mesh
        solver.mesh->resize_wake(vec_t.size()-1); // +1 for the initial pose

        // f32 t = 0.0f;
        // for (u64 i = 0; i < vec_dt.size(); i++) {
        //     FlowData flow{};
        //     solver.solve(flow);
        //     solver.mesh->shed_wake();
        //     solver.mesh->move(linalg::mul(displacement(t+vec_dt[i]), linalg::inverse(displacement(t))));
        //     t += vec_dt[i];
        // }
    }
 
    return 0;
}
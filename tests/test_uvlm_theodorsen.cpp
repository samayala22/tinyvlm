#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <functional> // std::function

#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_utils.hpp"

using namespace vlm;
using namespace linalg::ostream_overloads;

using tmatrix = linalg::alias::float4x4; // transformation matrix

class Kinematics {
    public:
    std::vector<std::function<tmatrix(f32 t)>> joints;
    Kinematics() = default;
    ~Kinematics() = default;

    void add(std::function<tmatrix(f32 t)>&& joint) {
        joints.push_back(std::move(joint));
    }

    tmatrix displacement(f32 t) {
        tmatrix result = linalg::identity;
        for (const auto& joint : joints) {
            result = linalg::mul(result, joint(t));
        }
        return result;
    }

    tmatrix relative_displacement(f32 t0, f32 t1) {
        return linalg::mul(displacement(t1), linalg::inverse(displacement(t0)));
    }

    f32 absolute_velocity(f32 t, const linalg::alias::float4& vertex) {
        return linalg::length((linalg::mul(relative_displacement(t, t+EPS_sqrt_f), vertex)-vertex)/EPS_sqrt_f);
    }
};

int main() {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_5x200.x"};
    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    // Define simulation length
    const f32 t_final = 10.0f;

    Kinematics kinematics{};

    // Define wing motion
    kinematics.add([](f32 t) {
        return linalg::translation_matrix(linalg::alias::float3{
            0.0f,
            0.0f,
            std::sin(0.2f * t)
        });
    });

    kinematics.add([](f32 t) {
        return linalg::translation_matrix(linalg::alias::float3{
            -1.0f*t,
            0.0f,
            0.0f
        });
    });

    for (const auto& [mesh_name, backend_name] : solvers) {
        UVLM solver{};
        solver.mesh = create_mesh(mesh_name);

        // Pre-calculate timesteps to determine wake size
        std::vector<f32> vec_t; // timesteps
        vec_t.push_back(0.0f);
        for (f32 t = 0.0f; t < t_final;) {
            const f32 dt = (0.5f * (solver.mesh->chord_length(0) + solver.mesh->chord_length(1))) / kinematics.absolute_velocity(t, {0.f, 0.f, 0.f, 1.f});
            t += dt;
            vec_t.push_back(t);
        }

        solver.mesh->resize_wake(vec_t.size()-1); // +1 for the initial pose
        solver.backend = create_backend(backend_name, *solver.mesh); // create after mesh has been resized

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
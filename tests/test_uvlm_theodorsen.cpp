#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <functional> // std::function

#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_data.hpp"
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

    linalg::alias::float4 velocity(f32 t, const linalg::alias::float4& vertex) {
        return (linalg::mul(relative_displacement(t, t+EPS_sqrt_f), vertex)-vertex)/EPS_sqrt_f;
    }

    f32 velocity_magnitude(f32 t, const linalg::alias::float4& vertex) {
        return linalg::length(velocity(t, vertex));
    }
};

int main() {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_2x8.x"};
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
        std::unique_ptr<Mesh> mesh = create_mesh(mesh_name);

        // Pre-calculate timesteps to determine wake size
        std::vector<f32> vec_t; // timesteps
        vec_t.push_back(0.0f);
        for (f32 t = 0.0f; t < t_final;) {
            const f32 dt = (0.5f * (mesh->chord_length(0) + mesh->chord_length(1))) / kinematics.velocity_magnitude(t, {0.f, 0.f, 0.f, 1.f});
            t += dt;
            vec_t.push_back(t);
        }

        mesh->resize_wake(vec_t.size()-1); // +1 for the initial pose
        std::unique_ptr<Backend> backend = create_backend(backend_name, *mesh); // create after mesh has been resized

        // Precompute the LHS since wing geometry is constant
        FlowData flow_dummy{0.0f, 0.0f, 1.0f, 1.0f};
        backend->compute_lhs(flow_dummy);
        backend->lu_factor();
        for (u64 i = 0; i < vec_t.size()-1; i++) {
            auto base_vertex = mesh->get_v0(0);
            auto base_velocity = kinematics.velocity(vec_t[i], {base_vertex[0], base_vertex[1], base_vertex[2], 1.0f});
            FlowData flow{linalg::alias::float3{base_velocity[0], base_velocity[1], base_velocity[2]}, 1.0f};
            backend->compute_rhs(flow);
            backend->add_wake_influence(flow);
            backend->lu_solve();
            backend->compute_delta_gamma();
            // compute d gamma / dt
            const f32 cl_steady = backend->compute_coefficient_cl(flow);
            // const f32 cl_unsteady = backend->compute_coefficient_unsteady_cl(flow); // TODO: implement
            // std::printf("t: %f, CL: %f\n", vec_t[i], cl_steady + cl_unsteady);
            mesh->move(kinematics.relative_displacement(vec_t[i], vec_t[i+1]));
            backend->shed_gamma();
        }
    }
 
    return 0;
}
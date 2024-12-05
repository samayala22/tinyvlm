#include <vector>
#include <string>
#include <functional> // std::function

#include "tinycombination.hpp"
#include "tinyad.hpp"
#include "tinypbar.hpp"

#include "vlm.hpp"
#include "vlm_types.hpp"
#include "vlm_kinematics.hpp"
#include "vlm_utils.hpp"

using namespace vlm;
using namespace linalg::ostream_overloads;

int main() {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_5x10.x"};
    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 5.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 k = 0.5; // reduced frequency
    const f32 omega = k * u_inf / b;
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods
    //const f32 t_final = 5.0f;

    KinematicsTree kinematics_tree;

    // Periodic heaving
    // const f32 amplitude = 0.1f; // amplitude of the wing motion
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    // });
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({0.f, 0.f, amplitude * fwd::sin(omega * t)});
    // });

    // Periodic pitching
    const f32 amplitude = 3.f; // amplitude in degrees
    auto fs = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    });
    auto pitch = kinematics_tree.add([=](const fwd::Float& t) {
        return rotation_matrix<fwd::Float>({0.25f, 0.0f, 0.0f},{0.0f, 1.0f, 0.0f}, to_radians(amplitude) * fwd::sin(omega * t));
    })->after(fs);
    
    // Sudden acceleration
    // const f32 alpha = to_radians(5.0f);
    // auto freestream = kinematics_tree.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({
    //         -u_inf*std::cos(alpha)*t,
    //         0.0f,
    //         -u_inf*std::sin(alpha)*t
    //     });
    // });

    for (const auto& [mesh_name, backend_name] : solvers) {
        Assembly assembly(fs);
        assembly.add(mesh_name, pitch);
        UVLM simulation{backend_name, {mesh_name}};
        simulation.run(assembly, t_final);
    }
    return 0;
}
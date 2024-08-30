#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <functional> // std::function

#include "tinycombination.hpp"
#include "tinyad.hpp"
#include "tinytimer.hpp"

#include "vlm.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_executor.hpp"
#include "vlm_kinematics.hpp"

// #define DEBUG_DISPLACEMENT_DATA

using namespace vlm;
using namespace linalg::ostream_overloads;

template<typename T>
void dump_buffer(std::ofstream& stream, T* start, T* end) {
    for (T* it = start; it != end; it++) {
        stream << *it << " ";
    }
    stream << "\n";
}

template<typename T>
void print_buffer(const T* start, u64 size) {
    std::cout << "[";
    for (u64 i = 0; i < size; i++) {
        std::cout << start[i] << ",";
    }
    std::cout << "]\n";
}

int main() {
    const u64 ni = 2;
    const u64 nj = 2;
    vlm::Executor::instance(1);
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_" + std::to_string(ni) + "x" + std::to_string(nj) + ".x"};
    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 1.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 0.5; // reduced frequency
    const f32 omega = k * 2.0f * u_inf / (2*b);
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods
    //const f32 t_final = 5.0f;

    Kinematics kinematics{};

    const f32 initial_angle = 0.0f;

    const auto initial_pose = rotation_matrix(
        linalg::alias::float3{0.0f, 0.0f, 0.0f}, // take into account quarter chord panel offset
        linalg::alias::float3{0.0f, 1.0f, 0.0f},
        to_radians(initial_angle)
    );

    // Periodic heaving
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    // });
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({0.f, 0.f, amplitude * fwd::sin(omega * t)});
    // });

    // Periodic pitching
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    // });
    // const f32 to_rad = PI_f / 180.0f;
    // kinematics.add([=](const fwd::Float& t) {
    //     return rotation_matrix<fwd::Float>({0.25f, 0.0f, 0.0f},{0.0f, 1.0f, 0.0f}, to_rad * fwd::sin(omega * t));
    // });
    
    // Sudden acceleration
    const f32 alpha = to_radians(5.0f);
    kinematics.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            -u_inf*std::cos(alpha)*t,
            0.0f,
            -u_inf*std::sin(alpha)*t
        });
    });

    for (const auto& [mesh_name, backend_name] : solvers) {
        UVLM simulation{backend_name, {mesh_name}};
        simulation.run({kinematics}, {initial_pose}, t_final);
    }

    return 0;
}
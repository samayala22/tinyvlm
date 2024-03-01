#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>

#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_utils.hpp"

using namespace vlm;

int main(int argc, char **argv) {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_5x200.x"};
    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    for (const auto& [mesh_name, backend_name] : solvers) {

        auto translation = [&](f32 t) -> linalg::alias::float4x4 {
            return linalg::translation_matrix(linalg::alias::float3{
                -1.0f*t,
                0.0f,
                std::sin(0.2f * t)
            });
        };

        // auto rotation = [&](f32 t) -> linalg::alias::float4x4 {
        //     return linalg::translation_matrix(linalg::alias::float3{
        //         -1.0f*t,
        //         0.0f,
        //         std::sin(0.2f * t)
        //     });
        // };

        auto wing_pose = [&](f32 t) -> linalg::alias::float4x4 {
            return translation(t);
        };

    }
 
    return 0;
}
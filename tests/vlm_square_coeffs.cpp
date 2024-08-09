#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>

#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_utils.hpp"

using namespace vlm;

float compute_analytical_cl(float alpha) {
    return 2.0f * vlm::PI_f * alpha;
}

int main(int  /*argc*/, char ** /*argv*/) {
    const std::vector<std::vector<std::string>> meshes = {
        {
            "../../../../mesh/infinite_rectangular_2x2_part0.x",
            "../../../../mesh/infinite_rectangular_2x2_part1.x"
        }
    };

    const std::vector<std::string> backends = get_available_backends();
    std::vector<f32> test_alphas = {0, 1, 2, 3, 4, 5, 10, 15};
    //std::vector<f32> test_alphas = {1.f};
    std::transform(test_alphas.begin(), test_alphas.end(), test_alphas.begin(), to_radians);

    auto solvers = tiny::make_combination(meshes, backends);
    for (const auto& [meshes_names, backend_name] : solvers) {
        std::printf(">>> BACKEND: %s\n", backend_name.get().c_str());

        VLM simulation{backend_name, meshes_names};

        for (u64 i = 0; i < test_alphas.size(); i++) {
            const FlowData flow{test_alphas[i], 0.0f, 1.0f, 1.0f};
            const auto coeffs = simulation.run(flow);
            const f32 analytical_cl = compute_analytical_cl(flow.alpha);
            const f32 cl_aerr = std::abs(coeffs.cl - analytical_cl);
            const f32 cl_rerr = analytical_cl < EPS_f ? 0.f : cl_aerr / analytical_cl;
            std::printf(">>> Alpha: %.1f | CL = %.7f CD = %.7f CMx = %.6f CMy = %.6f CMz = %.6f\n", to_degrees(flow.alpha), coeffs.cl, coeffs.cd, coeffs.cm.x, coeffs.cm.y, coeffs.cm.z);
            std::printf(">>> Analytical CL: %.7f | Abs Error: %.3E | Relative Error: %.5f%% \n", analytical_cl, cl_aerr, cl_rerr*100.f);
            if (cl_rerr > 0.01f) return 1;
        }
    }
    return 0;
}
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>

#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_utils.hpp"

using namespace vlm;

// area of whole wing
float s_ref(float a, float b) {
    return 0.5f * vlm::PI_f * a * b;
}

// wing aspect ratio
float aspect_ratio(float a, float b) {
    return (2*b)*(2*b) / s_ref(a, b);
}

float compute_analytical_cl(float alpha, float a, float b) {
    return 2.0f * vlm::PI_f * alpha / (1.0f + 2.0f/aspect_ratio(a, b));
}

float compute_analytical_cd(float cl, float a, float b) {
    return cl*cl / (vlm::PI_f * aspect_ratio(a, b));
}

float compute_analytical_gamma(float y, float a, float b, float alpha) {
    const float gamma0 = compute_analytical_cl(alpha, a, b) * 1.0f * s_ref(a, b) / (vlm::PI_f * b);
    const float ratio = y / b;
    return gamma0 * std::sqrt(1.0f - ratio*ratio);
}

int main(int  /*argc*/, char ** /*argv*/) {
    const float a = 1.0f; // wing chord root
    const float b = 5.0f; // half wing span

    const std::vector<std::string> meshes = {"../../../../mesh/elliptic_45x45.x"};
    const std::vector<std::string> backends = get_available_backends();
    
    std::vector<f32> test_alphas = {0, 1, 2, 3, 4, 5, 10, 15};
    std::transform(test_alphas.begin(), test_alphas.end(), test_alphas.begin(), to_radians);

    auto solvers = tiny::make_combination(meshes, backends);
    for (const auto& [mesh_name, backend_name] : solvers) {
        std::printf("\nBACKEND: %s\n", backend_name.get().c_str());
        std::printf("MESH: %s\n", mesh_name.get().c_str());

        VLM simulation{backend_name, {mesh_name}};

        std::printf("\n|    Alpha   |     CL     |     CD     |    CMx     |    CMy     |    CMz     |  CL Error   |  CD Error   |\n");
        std::printf("|------------|------------|------------|------------|------------|------------|-------------|-------------|\n");

        for (u64 i = 0; i < test_alphas.size(); i++) {
            const FlowData flow{test_alphas[i], 0.0f, 1.0f, 1.0f};
            const auto coeffs = simulation.run(flow);
            const f32 analytical_cl = compute_analytical_cl(flow.alpha, a, b);
            const f32 analytical_cd = compute_analytical_cd(analytical_cl, a, b);
            const f32 cl_aerr = std::abs(coeffs.cl - analytical_cl);
            const f32 cl_rerr = analytical_cl < EPS_f ? 0.f : cl_aerr / analytical_cl;
            const f32 cd_aerr = std::abs(coeffs.cd - analytical_cd);
            const f32 cd_rerr = analytical_cd < EPS_f ? 0.f : cd_aerr / analytical_cd;
            // std::printf(">>> Alpha: %.1f | CL = %.7f CD = %.7f CMx = %.6f CMy = %.6f CMz = %.6f\n", to_degrees(flow.alpha), coeffs.cl, coeffs.cd, coeffs.cm.x, coeffs.cm.y, coeffs.cm.z);
            // std::printf(">>> Analytical CL: %.7f | Abs Error: %.3E | Relative Error: %.5f%% \n", analytical_cl, cl_aerr, cl_rerr*100.f);
            // std::printf(">>> Analytical CD: %.7f | Abs Error: %.3E | Relative Error: %.5f%% \n", analytical_cd, cd_aerr, cd_rerr*100.f);
            std::printf("| %10.1f | %10.6f | %10.7f | %10.6f | %10.6f | %10.6f | %10.3f%% | %10.3f%% |\n",
                to_degrees(flow.alpha),
                coeffs.cl,
                coeffs.cd,
                coeffs.cm.x,
                coeffs.cm.y,
                coeffs.cm.z,
                cl_rerr * 100.0f,
                cd_rerr * 100.0f
            );
            
            if (cl_rerr > 0.03f || cd_rerr > 0.03f) return 1;
        }
    }
    return 0;
}
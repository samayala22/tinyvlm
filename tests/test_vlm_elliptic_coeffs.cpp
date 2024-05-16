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

// bool elliptic_convergence() {
//     tiny::Config cfg("../../../../config/elliptic.vlm");

//     vlm::VLM vlm(cfg);

//     std::vector<int> dimensions = {
//         16, 32, 45, 64, 90, 128
//     };

//     const float a = 1.0f; // wing chord root
//     const float b = 5.0f; // half wing span

//     std::vector<double> norm_l1;
//     std::vector<double> norm_l2;
//     std::vector<double> norm_linf;

//     const float alpha = 0.1f; // degrees

//     for (const auto& dim : dimensions) {
//         std::string filename = std::format("../../../../mesh/elliptic_{}x{}.x", dim, dim);
//         vlm.mesh.io_read(filename);
//         vlm.init();
//         vlm::Solver solver(vlm.mesh, vlm.data, cfg);
//         solver.run(alpha);

//         double l1 = 0.0f;
//         double l2 = 0.0f;
//         double linf = 0.0f;
//         int begin = (vlm.mesh.nc - 1) * vlm.mesh.ns;
//         int end = vlm.mesh.nc * vlm.mesh.ns;
//         // loop over last row of panels
//         for (int i = begin; i < end; i++) {
//             const float y = vlm.mesh.colloc.y[i];
//             const float gamma = vlm.data.gamma[i];
//             const float gamma_analytical = analytical_gamma(y, a, b, alpha);
//             const double error = std::abs((gamma - gamma_analytical) / (gamma_analytical + 1e-7f));
//             std::printf("y: %f, gamma: %f, gamma_analytical: %f, error: %f \n", y, gamma, gamma_analytical, error);

//             l1 += error;
//             l2 += error * error;
//             linf = std::max(linf, error);
//         }
//         l1 /= (end - begin);
//         l2 = std::sqrt(l2) / (end - begin);
//         std::printf("L1: %f, L2: %f, Linf: %f\n", l1, l2, linf);

//         norm_l1.push_back(l1);
//         norm_l2.push_back(l2);
//         norm_linf.push_back(linf);
//     }

//     double order_l1 = 0.0f;
//     double order_l2 = 0.0f;
//     double order_linf = 0.0f;

//     auto order = [=](double norm0, double norm1, float dim0, float dim1) {
//         return std::log(norm0 / norm1) / std::log((b/dim0)/(b/dim1));
//     };

//     for (int i = 0; i < dimensions.size() - 1; i++) {
//         order_l1 += order(norm_l1[i], norm_l1[i+1], dimensions[i], dimensions[i+1]);
//         order_l2 += order(norm_l2[i], norm_l2[i+1], dimensions[i], dimensions[i+1]);
//         order_linf += order(norm_linf[i], norm_linf[i+1], dimensions[i], dimensions[i+1]);
//     }
//     order_l1 /= (dimensions.size() - 1);
//     order_l2 /= (dimensions.size() - 1);
//     order_linf /= (dimensions.size() - 1);
//     std::printf("Order L1: %f, Order L2: %f, Order Linf: %f\n", order_l1, order_l2, order_linf);
//     return 0;
// }

int main(int argc, char **argv) {
    const float a = 1.0f; // wing chord root
    const float b = 5.0f; // half wing span

    const std::vector<std::string> meshes = {"../../../../mesh/elliptic_64x64.x"};
    const std::vector<std::string> backends = get_available_backends();
    std::vector<f32> test_alphas = {0, 1, 2, 3, 4, 5, 10, 15};
    std::transform(test_alphas.begin(), test_alphas.end(), test_alphas.begin(), to_radians);

    auto solvers = tiny::make_combination(meshes, backends);
    for (const auto& [mesh_name, backend_name] : solvers) {
        LinearVLM solver{};
        solver.mesh = create_mesh(mesh_name);
        solver.backend = create_backend(backend_name, *solver.mesh);

        for (u64 i = 0; i < test_alphas.size(); i++) {
            const FlowData flow{test_alphas[i], 0.0f, 1.0f, 1.0f};
            const auto coeffs = solver.solve(flow);
            const f32 analytical_cl = compute_analytical_cl(flow.alpha, a, b);
            const f32 analytical_cd = compute_analytical_cd(analytical_cl, a, b);
            const f32 cl_aerr = std::abs(coeffs.cl - analytical_cl);
            const f32 cl_rerr = analytical_cl < EPS_f ? 0.f : cl_aerr / analytical_cl;
            const f32 cd_aerr = std::abs(coeffs.cd - analytical_cd);
            const f32 cd_rerr = analytical_cd < EPS_f ? 0.f : cd_aerr / analytical_cd;
            std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", to_degrees(flow.alpha), coeffs.cl, coeffs.cd, coeffs.cm.x, coeffs.cm.y, coeffs.cm.z);
            std::printf(">>> Analytical CL: %.6f | Abs Error: %.3E | Relative Error: %.5f%% \n", analytical_cl, cl_aerr, cl_rerr*100.f);
            std::printf(">>> Analytical CD: %.6f | Abs Error: %.3E | Relative Error: %.5f%% \n", analytical_cd, cd_aerr, cd_rerr*100.f);
            if (cl_rerr > 0.03f || cd_rerr > 0.03f) return 1;
        } 
    }
    return 0;
}
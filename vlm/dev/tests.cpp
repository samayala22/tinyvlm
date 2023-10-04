#include "vlm.hpp"
#include "parser.hpp"
#include "vlm_types.hpp"
#include "vlm_solver.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <format>
#include <cstdio>

float s_ref(float a, float b) {
    return 0.5f * vlm::PI_f * a * b;
}

float aspect_ratio(float a, float b) {
    return (2*b)*(2*b) / s_ref(a, b);
}

float analytical_cl(float alpha, float a, float b) {
    return 2.0f * vlm::PI_f * alpha / (1.0f + 2.0f/aspect_ratio(a, b));
}

float analytical_gamma(float y, float a, float b, float alpha) {
    const float gamma0 = analytical_cl(alpha, a, b) * 1.0f * s_ref(a, b) / (vlm::PI_f * b);
    const float ratio = y / b;
    return gamma0 * std::sqrt(1.0f - ratio*ratio);
}

bool elliptic_convergence() {
    tiny::Config cfg("../../../../config/elliptic.vlm");

    vlm::VLM vlm(cfg);

    std::vector<int> dimensions = {
        64, 128
    };

    const float a = 1.0f; // wing chord root
    const float b = 5.0f; // single wing span

    std::vector<float> norm_l1;
    std::vector<float> norm_l2;
    std::vector<float> norm_linf;

    const float alpha = 0.1f * vlm::PI_f / 180.0f;

    for (const auto& dim : dimensions) {
        std::string filename = std::format("../../../../mesh/elliptic_{}x{}.xyz", dim, dim);
        vlm.mesh.io_read(filename);
        vlm.init();
        vlm::Solver solver(vlm.mesh, vlm.data, cfg);
        solver.run(alpha);

        float l1 = 0.0f;
        float l2 = 0.0f;
        float linf = 0.0f;
        int begin = (vlm.mesh.nc - 1) * vlm.mesh.ns;
        int end = vlm.mesh.nc * vlm.mesh.ns;
        // loop over last row of panels
        for (int i = begin; i < end; i++) {
            const float y = vlm.mesh.colloc.y[i];
            const float gamma = vlm.data.gamma[i];
            const float gamma_analytical = analytical_gamma(y, a, b, alpha);
            const float error = std::abs(gamma - gamma_analytical);
            l1 += error;
            l2 += error * error;
            linf = std::max(linf, error);
        }
        l1 /= (end - begin);
        l2 = std::sqrt(l2) / (end - begin);
        std::printf("L1: %f, L2: %f, Linf: %f\n", l1, l2, linf);

        norm_l1.push_back(l1);
        norm_l2.push_back(l2);
        norm_linf.push_back(linf);
    }

    float order_l1 = 0.0f;
    float order_l2 = 0.0f;
    float order_linf = 0.0f;
    auto order = [=](float norm0, float norm1, float dim0, float dim1) {
        return std::log(norm0 / norm1) / std::log((b/dim0)/(b/dim1));
    };
    for (int i = 0; i < dimensions.size() - 1; i++) {
        order_l1 += order(norm_l1[i], norm_l1[i+1], dimensions[i], dimensions[i+1]);
        order_l2 += order(norm_l2[i], norm_l2[i+1], dimensions[i], dimensions[i+1]);
        order_linf += order(norm_linf[i], norm_linf[i+1], dimensions[i], dimensions[i+1]);
    }
    order_l1 /= (dimensions.size() - 1);
    order_l2 /= (dimensions.size() - 1);
    order_linf /= (dimensions.size() - 1);
    std::printf("Order L1: %f, Order L2: %f, Order Linf: %f\n", order_l1, order_l2, order_linf);
    return 0;
}

int main(int argc, char **argv) {
    try {
        std::printf(">>> Elliptic convergence | %d", elliptic_convergence());
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
    return 0;
}
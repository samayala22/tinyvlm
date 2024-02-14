#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <utility>

#include "tinyinterpolate.hpp"
#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_utils.hpp"

using namespace vlm;

struct LiftCurveFunctor {
    virtual f32 operator()(f32 alpha) const = 0;
};

struct SpallartLiftCurve : public LiftCurveFunctor {
    const f32 cl_0, a0, a1, cl_a0, cl_a1;
    SpallartLiftCurve(f32 cl_0_, f32 a0_, f32 a1_, f32 cl_a0_, f32 cl_a1_):
        cl_0(cl_0_), a0(a0_), a1(a1_), cl_a0(cl_a0_), cl_a1(cl_a1_) {}
    inline f32 operator()(f32 alpha) const override {
        return cl_a0 * alpha + 0.5f * (cl_0 - cl_a1 * alpha) * (1.f + std::erf((alpha - a0) / a1));
    }
};

struct ThinAirfoilPolarLiftCurve : public LiftCurveFunctor{
    ThinAirfoilPolarLiftCurve() {}
    inline f32 operator()(f32 alpha) const override {
        return 2.f * PI_f * alpha;
    }
};

template<typename T>
void linspace(T start, T end, u32 n, std::vector<T>& out) {
    out.resize(n);
    T step = (end - start) / (n - 1);
    for (u32 i = 0; i < n; i++) {
        out[i] = start + i * step;
    }
}

template<typename T>
void write_vector_pair(const std::string& filename, const std::vector<T>& vec1, const std::vector<T>& vec2) {
    assert(vec1.size() == vec2.size());
    std::ofstream outFile(filename + ".dat");

    // Check if file is open
    if (!outFile.is_open()) throw std::runtime_error("Failed to open file: " + filename);

    const u64 n = vec1.size();
    outFile << n << '\n';
    for (u64 i = 0; i < n; i++) {
        outFile << vec1[i] << ' ' << vec2[i] << '\n';
    }
    outFile.close();
}

int main(int argc, char** argv) {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_5x200.x"};
    const std::vector<std::string> backends = {"cpu"};
    std::vector<std::pair<std::string, std::unique_ptr<LiftCurveFunctor>>> lift_curves;
    lift_curves.emplace_back(std::make_pair("spallart1", std::make_unique<SpallartLiftCurve>(1.2f, 0.28f, 0.02f, 2.f*PI_f, 2.f*PI_f)));
    // lift_curves.emplace_back(std::make_pair("spallart2", std::make_unique<SpallartLiftCurve>(0.72f, 0.28f, 0.04f, 2.f*PI_f, 1.5f*PI_f)));
    // lift_curves.emplace_back(std::make_pair("polar", std::make_unique<ThinAirfoilPolarLiftCurve>()));
    
    std::vector<f32> test_alphas = {0, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::transform(test_alphas.begin(), test_alphas.end(), test_alphas.begin(), to_radians);
    std::vector<f32> test_cl(test_alphas.size());

    std::vector<f32> db_alphas;
    linspace(to_radians(0.f), to_radians(20.f), 100, db_alphas);
    const std::vector<f32> db_dummy(db_alphas.size(), 0.0f);
    std::vector<f32> db_cl(db_alphas.size()); // modified by lift function

    auto solvers = tiny::make_combination(meshes, backends);

    for (const auto& [mesh_name, backend_name] : solvers) {
        NonLinearVLM solver{1e-5f, 100};
        solver.mesh = create_mesh(mesh_name);
        solver.backend = create_backend(backend_name, *solver.mesh);

        for (const auto& lift_curve : lift_curves) {
            std::transform(db_alphas.begin(), db_alphas.end(), db_cl.begin(), [&lift_curve](float alpha){ return (*lift_curve.second)(alpha); });
            write_vector_pair(lift_curve.first + "_analytical_cl", db_alphas, db_cl);

            Database db;
            db.profiles.emplace_back(
                WingProfile<tiny::AkimaInterpolator<f32>>(
                    db_alphas,
                    db_cl,
                    db_dummy, 
                    db_dummy,
                    db_dummy,
                    db_dummy               
                )
            );
            db.profiles_pos.emplace_back(0.0f);
            
            for (u32 i = 0; i < test_alphas.size(); i++) {
                const FlowData flow{test_alphas[i], 0.0f, 1.0f, 1.0f};
                auto coeffs = solver.solve(flow, db);
                test_cl[i] = coeffs.cl;
                std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", to_degrees(test_alphas[i]), coeffs.cl, coeffs.cd, coeffs.cm.x, coeffs.cm.y, coeffs.cm.z);
                const f32 analytical_cl = (*lift_curve.second)(flow.alpha);
                const f32 abs_error = std::abs(coeffs.cl - analytical_cl);
                const f32 rel_error = abs_error / (analytical_cl + std::numeric_limits<f32>::epsilon());
                std::printf(">>> Analytical: %.6f | Abs Error: %.3E | Relative Error: %.5f%% \n", analytical_cl, abs_error, rel_error*100.f);
                if (rel_error > 0.01f) return 1; // Failure
            }
            write_vector_pair(lift_curve.first + "_nonlinear_cl", test_alphas, test_cl);
        }
    }

    return 0; // Success
}
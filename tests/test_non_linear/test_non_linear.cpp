#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <cstdio>

#include "vlm.hpp"
#include "tinycombination.hpp"

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

f32 to_radians(f32 degrees) {
    return degrees * PI_f / 180.0f;
}

// SpallartLiftCurve lift_curve(1.2f, 0.28f, 0.02f, 2.f*PI_f, 2.f*PI_f);
// ThinAirfoilPolarLiftCurve lift_curve{};

int main(int argc, char** argv) {
    std::vector<std::string> meshes = {"../../../../tests/test_non_linear/meshes/infinite_rectangular_5x1000.x"};
    std::vector<std::string> backends = {"avx2"};
    std::vector<std::unique_ptr<LiftCurveFunctor>> lift_curves;
    lift_curves.emplace_back(std::make_unique<SpallartLiftCurve>(1.2f, 0.28f, 0.02f, 2.f*PI_f, 2.f*PI_f));
    lift_curves.emplace_back(std::make_unique<SpallartLiftCurve>(0.72f, 0.28f, 0.04f, 2.f*PI_f, 1.5f*PI_f));
    lift_curves.emplace_back(std::make_unique<ThinAirfoilPolarLiftCurve>());
    std::vector<f32> alphas = {0, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    std::vector<f32> db_alphas;
    linspace(to_radians(0.f), to_radians(20.f), 100, db_alphas);

    auto combinations = tiny::make_combination(meshes, backends, lift_curves);

    for (const auto& meshfile : meshes) {
        // TODO: find a way to clean this up
        Mesh mesh{meshfile};
        Data data{};
        mesh.compute_connectivity();
        mesh.compute_metrics_wing();
        if (data.c_ref == 0.0f) data.c_ref = mesh.chord_mean(0, mesh.ns+1);
        if (data.s_ref == 0.0f) data.s_ref = mesh.panels_area_xy(0,0, mesh.nc, mesh.ns);
        data.alloc(mesh.nc*mesh.ns);

        for (const auto& backend_name : backends) {
            auto backend = create_backend(backend_name, mesh, data);
            
            for (const auto& lift_curve : lift_curves) {
                
            }
        }

    }

    return 0; // Success
}
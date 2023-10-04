#include "vlm.hpp"
#include "vlm_solver.hpp"

#include <oneapi/tbb/global_control.h>

using namespace vlm;

VLM::VLM(tiny::Config& cfg) : mesh(cfg), data(cfg) {};

void VLM::init() {
    io.read_mesh(mesh);
    data.alloc(mesh.nc*mesh.ns);
}

void VLM::preprocess() {
    mesh.compute_connectivity();
    mesh.compute_metrics_wing();
}

void VLM::solve(tiny::Config& cfg) {
    Solver solver(mesh, data, cfg);
    std::vector<f32> alphas = cfg().section("solver").get_vector<f32>("alphas", {0.0f});
    //tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
    for (auto alpha : alphas) {
        solver.run(alpha);
    }
}
#include "vlm.hpp"
#include "vlm_solver.hpp"

using namespace vlm;

void VLM::init() {
    io.read_config(config);
    io.read_mesh(mesh);
    data.alloc(mesh.nc, mesh.ns);
}

void VLM::preprocess() {
    mesh.compute_connectivity();
    mesh.compute_metrics_wing();
}

void VLM::solve() {
    Solver solver(mesh, data, io, config);
    for (auto alpha : config.alphas) {
        solver.run(alpha);
    }
}
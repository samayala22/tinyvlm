#include "vlm.hpp"
#include "vlm_solver.hpp"

#include <oneapi/tbb/global_control.h>

using namespace vlm;

void VLM::init() {
    io.read_config(config);
    io.read_mesh(mesh);
    data.alloc(mesh.nc, mesh.ns);
    data.s_ref = config.S_ref;
    data.ref_pt = config.ref_pt;
}

void VLM::preprocess() {
    mesh.compute_connectivity();
    mesh.compute_metrics_wing();
}

void VLM::solve() {
    Solver solver(mesh, data, io, config);
    //tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
    for (auto alpha : config.alphas) {
        solver.run(alpha);
    }
}
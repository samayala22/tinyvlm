#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"

namespace vlm {

struct Solver {
    Mesh& mesh;
    Data& data;
    IO& io;
    Config& config;

    std::vector<f32> lhs = {}; // influence matrix (square non-symmetric, col major)
    std::vector<f32> rhs = {}; // right hand side (-Qn)

    void run(const f32 alpha);
    void reset();
    void compute_lhs();
    void compute_rhs();
    void solve();
    void compute_forces();
    Solver(Mesh& mesh, Data& data, IO& io, Config& config);
};

} // namespace vlm
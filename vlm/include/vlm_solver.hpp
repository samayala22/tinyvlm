#pragma once

#include "vlm_types.hpp"
#include "vlm_fwd.hpp"

namespace vlm {

struct Solver {
    Mesh& mesh;
    Data& data;

    tiny::vector<f32, 64> lhs; // influence matrix (square non-symmetric, col major)
    tiny::vector<f32, 64> rhs; // right hand side

    void run(const f32 alpha_deg);
    void reset();
    void compute_lhs();
    void compute_rhs();
    void solve();
    void compute_forces();
    void compute_delta_gamma();
    Solver(Mesh& mesh, Data& data, tiny::Config& cfg);
};

} // namespace vlm
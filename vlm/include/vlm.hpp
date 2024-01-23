#pragma once

#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"

namespace vlm {

struct VLM {
    Mesh mesh;
    Data data;

    void init();
    void solve(tiny::Config& cfg);
    void solve_nonlinear(tiny::Config& cfg);
    VLM() = default;
    VLM(tiny::Config& cfg);
};

} // namespace vlm

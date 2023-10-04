#pragma once

#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_io.hpp"
#include "vlm_types.hpp"

namespace vlm {

struct VLM {
    Mesh mesh;
    Data data;
    IO io;

    void init();
    void preprocess();
    void solve(tiny::Config& cfg);
    VLM() = default;
    VLM(tiny::Config& cfg);
};

} // namespace vlm

#pragma once

#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_io.hpp"
#include "vlm_types.hpp"
#include "vlm_config.hpp"

namespace vlm {

struct VLM {
    Mesh mesh;
    Data data;
    IO io;
    Config config;

    void init();
    void preprocess();
    void solve();
};

} // namespace vlm

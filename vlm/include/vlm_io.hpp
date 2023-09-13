#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "tinyconfig.hpp"

namespace vlm {

struct IO {
    std::string filename_mesh;
    std::string filename_result;
    std::string filename_config;

    void read_config(Config& config);
    void read_mesh(Mesh& mesh);
    void write_result(Mesh& mesh, Data& data);
};

}
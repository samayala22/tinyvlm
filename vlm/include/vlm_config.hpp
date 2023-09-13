#pragma once

#include "vlm_types.hpp"

namespace vlm {

struct Config {
    std::vector<f32> alphas = {};
    bool wake_included = false;
};

}
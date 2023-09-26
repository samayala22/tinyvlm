#pragma once

#include "vlm_types.hpp"

namespace vlm {

struct Config {
    std::vector<f32> alphas = {};
    bool wake_included = false;
    f32 S_ref = 0.0f;
    Vec3 ref_pt = {0.25f, 0.0f, 0.0f};
};

}
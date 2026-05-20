#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "vlm_types.hpp"
#include "vlm_memory.hpp"

namespace vlm {

using SurfDims = std::pair<i64, i64>; // nc, ns

class MeshIO {
public:
    MeshIO(const std::string& format);
    SurfDims get_dims(const std::string& filename) const;
    void read(const std::string& filename, const TensorView3fH& vertices, bool qc = true) const;
    void read(const std::string& filename, const TensorView3dH& vertices, bool qc = true) const;
};

} // namespace vlm

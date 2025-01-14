#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "vlm_types.hpp"
#include "vlm_memory.hpp"

#include "tinyfwd.hpp"

namespace vlm {

using SurfDims = std::pair<i64, i64>; // nc, ns

class MeshFile {
public:
    virtual ~MeshFile() = default;
    virtual SurfDims get_dims(std::ifstream& file) const = 0;
    [[deprecated]] virtual void read(std::ifstream& file, const TensorView3D<Location::Host>& vertices) const = 0;
    virtual void read2(std::ifstream& file, const TensorView3D<Location::Host>& vertices) const = 0;
};

class MeshIO {
public:
    MeshIO(const std::string& format);
    SurfDims get_dims(const std::string& filename) const;
    void read(const std::string& filename, const TensorView3D<Location::Host>& vertices, bool legacy=true) const;

private:
    std::unique_ptr<MeshFile> _file;
};

} // namespace vlm

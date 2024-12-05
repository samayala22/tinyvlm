#include "vlm_mesh.hpp"
#include "linalg.h"
#include "vlm_types.hpp"

#include "tinyconfig.hpp"
#include "tinytimer.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <memory>

using namespace vlm;

void mesh_quarterchord(const TensorView3D<Location::Host>& v) {
    for (i64 j = 0; j < v.shape(1) - 1; j++) {
        for (i64 i = 0; i < v.shape(0); i++) {
            v(i, j, 0) = 0.75f * v(i, j, 0) + 0.25f * v(i, j+1, 0);
            v(i, j, 1) = 0.75f * v(i, j, 1) + 0.25f * v(i, j+1, 1);
            v(i, j, 2) = 0.75f * v(i, j, 2) + 0.25f * v(i, j+1, 2);
        }
    }

    // Trailing edge vertices
    const i64 j = v.shape(1) - 2;
    for (i64 i = 0; i < v.shape(0); i++) {
        v(i, j+1, 0) = (4.f/3.f) * v(i, j+1, 0) - (1.f/3.f) * v(i, j, 0);
        v(i, j+1, 1) = (4.f/3.f) * v(i, j+1, 1) - (1.f/3.f) * v(i, j, 1);
        v(i, j+1, 2) = (4.f/3.f) * v(i, j+1, 2) - (1.f/3.f) * v(i, j, 2);
    }
}

class Plot3DFile : public MeshFile {
public:
    SurfDims get_dims(std::ifstream& f) const override {
        i64 ni, nj, nk;
        i64 blocks;
        f >> blocks;
        if (blocks != 1) {
            throw std::runtime_error("Only single block plot3d mesh is supported");
        }
        f >> ni >> nj >> nk;
        if (nk != 1) {
            throw std::runtime_error("Only 2D plot3d mesh is supported");
        }
        return {ni-1, nj-1};
    }

    void read(std::ifstream& f, const TensorView3D<Location::Host>& vertices) const override {
        i64 ni, nj, nk;
        i64 blocks;
        f32 x, y, z;
        f >> blocks;
        f >> ni >> nj >> nk;
        assert(vertices.shape(0) == nj);
        assert(vertices.shape(1) == ni);
        assert(vertices.shape(2) == 4);
        
        for (i64 j = 0; j < nj; j++) {
            for (i64 i = 0; i < ni; i++) {
                f >> x;
                vertices(j, i, 0) = x;
            }
        }
        for (i64 j = 0; j < nj; j++) {
            for (i64 i = 0; i < ni; i++) {
                f >> y;
                vertices(j, i, 1) = y;
            }
        }
        for (i64 j = 0; j < nj; j++) {
            for (i64 i = 0; i < ni; i++) {
                f >> z;
                vertices(j, i, 2) = z;
            }
        }

        // TODO: this validation is not robust enough when eventually we will be using multiple bodies
        // Validation
        // const f32 eps = std::numeric_limits<f32>::epsilon();
        // if (std::abs(m.v.x[0]) != eps || std::abs(m.v.y[0]) != eps) {
        //     throw std::runtime_error("First vertex of plot3d mesh must be at origin");
        // }
        const f32 first_y = vertices(0,0,1);
        for (i64 i = 1; i < ni; i++) {
            if ( vertices(0, i, 1) != first_y) {
                throw std::runtime_error("Mesh vertices should be ordered in chordwise direction"); // todo get rid of throw
            }
        }
    }
};

MeshIO::MeshIO(const std::string& format) {
    if (format == "plot3d") {
        _file = std::make_unique<Plot3DFile>();
    } else {
        throw std::runtime_error("Unsupported file format");
    }
}

SurfDims MeshIO::get_dims(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open mesh file");
    }
    auto dims = _file->get_dims(file);
    std::printf("MESH: %s (%llu x %llu)\n", filename.c_str(), dims.first, dims.second);
    return dims;
}

void MeshIO::read(const std::string& filename, const TensorView3D<Location::Host>& vertices) const {
    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Failed to open mesh file");
    }
    _file->read(file_stream, vertices);
    mesh_quarterchord(vertices);
}
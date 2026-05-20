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

template<typename T>
void mesh_quarterchord(const TensorView<T, 3, Location::Host>& v) {
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

SurfDims plot3d_get_dims(std::ifstream& f) {
    i64 ni, nj, nk;
    i64 blocks;
    f >> blocks;
    if (blocks != 1) {
        std::printf("Only single block plot3d mesh is supported\n");
        throw std::runtime_error("Only single block plot3d mesh is supported");
    }
    f >> ni >> nj >> nk;
    if (nk != 1) {
        std::printf("Only 2D plot3d mesh is supported\n");
        throw std::runtime_error("Only 2D plot3d mesh is supported");
    }
    return {ni-1, nj-1};
}

template<typename T>
void plot3d_read(std::ifstream& f, const TensorView<T, 3, Location::Host>& vertices) {
    i64 ni, nj, nk;
    i64 blocks;
    T x, y, z;
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
    const T first_y = vertices(0,0,1);
    for (i64 i = 1; i < ni; i++) {
        if ( vertices(0, i, 1) != first_y) {
            throw std::runtime_error("Mesh vertices should be ordered in chordwise direction"); // todo get rid of throw
        }
    }
}

MeshIO::MeshIO(const std::string& format) {
}

SurfDims MeshIO::get_dims(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::printf("Failed to open mesh file\n");
        throw std::runtime_error("");
    }
    auto dims = plot3d_get_dims(file);
    std::printf("MESH: %s (%llu x %llu)\n", filename.c_str(), dims.first, dims.second);
    return dims;
}

template<typename T>
void read_impl(const std::string& filename, const TensorView<T, 3, Location::Host>& vertices, bool qc) {
    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Failed to open mesh file");
    }
    plot3d_read(file_stream, vertices);
    if (qc) mesh_quarterchord(vertices);
}

void MeshIO::read(const std::string& filename, const TensorView3fH& vertices, bool qc) const {
    read_impl(filename, vertices, qc);
}

void MeshIO::read(const std::string& filename, const TensorView3dH& vertices, bool qc) const {
    read_impl(filename, vertices, qc);
}
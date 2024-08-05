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

void mesh_quarterchord(View<f32, SingleSurface>& vertices) {
    f32* vx = vertices.ptr + 0 * vertices.layout.stride();
    f32* vy = vertices.ptr + 1 * vertices.layout.stride();
    f32* vz = vertices.ptr + 2 * vertices.layout.stride();


    for (u64 i = 0; i < vertices.layout.surface().nc - 1; i++) {
        for (u64 j = 0; j < vertices.layout.surface().ns; j++) {
            const u64 v0 = (i+0) * vertices.layout.surface().ns + j;
            const u64 v3 = (i+1) * vertices.layout.surface().ns + j;

            vx[v0] = 0.75f * vx[v0] + 0.25f * vx[v3];
            vy[v0] = 0.75f * vy[v0] + 0.25f * vy[v3];
            vz[v0] = 0.75f * vz[v0] + 0.25f * vz[v3];
        }
    }

    // Trailing edge vertices
    const u64 i = vertices.layout.surface().nc - 2;
    for (u64 j = 0; j < vertices.layout.surface().ns; j++) {
        const u64 v0 = (i+0) * vertices.layout.surface().ns + j;
        const u64 v3 = (i+1) * vertices.layout.surface().ns + j;
        
        vx[v3] = (4.f/3.f) * vx[v3] - (1.f/3.f) * vx[v0];
        vy[v3] = (4.f/3.f) * vy[v3] - (1.f/3.f) * vy[v0];
        vz[v3] = (4.f/3.f) * vz[v3] - (1.f/3.f) * vz[v0];
    }
}

class Plot3DFile : public MeshFile {
public:
    SurfDims get_dims(std::ifstream& f) const override {
        u64 ni, nj, nk, blocks;
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

    void read(std::ifstream& f, View<f32, SingleSurface>& vertices) const override {
        u64 ni, nj, nk, blocks;
        f32 x, y, z;
        f >> blocks;
        f >> ni >> nj >> nk;
        assert(vertices.layout.surface().ns == nj);
        assert(vertices.layout.surface().nc == ni);
        assert(vertices.layout.dim() == 4);

        const u64 nb_vertices = ni * nj;
        
        for (u64 j = 0; j < nj; j++) {
            for (u64 i = 0; i < ni; i++) {
                f >> x;
                vertices[nb_vertices * 0 + nj*i + j] = x;
            }
        }
        for (u64 j = 0; j < nj; j++) {
            for (u64 i = 0; i < ni; i++) {
                f >> y;
                vertices[nb_vertices * 1 + nj*i + j] = y;
            }
        }
        for (u64 j = 0; j < nj; j++) {
            for (u64 i = 0; i < ni; i++) {
                f >> z;
                vertices[nb_vertices * 2 + nj*i + j] = z;
            }
        }

        // TODO: this validation is not robust enough when eventually we will be using multiple bodies
        // Validation
        // const f32 eps = std::numeric_limits<f32>::epsilon();
        // if (std::abs(m.v.x[0]) != eps || std::abs(m.v.y[0]) != eps) {
        //     throw std::runtime_error("First vertex of plot3d mesh must be at origin");
        // }
        const f32 first_y = vertices[nb_vertices * 1 + 0];
        for (u64 i = 1; i < ni; i++) {
            if ( vertices[nb_vertices * 1 + i*nj] != first_y) {
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
    std::cout << "Number of panels: " << dims.first * dims.second << "\n";
    std::cout << "nc: " << dims.first << "\n";
    std::cout << "ns: " << dims.second << "\n";
    return dims;
}

void MeshIO::read(const std::string& filename, View<f32, SingleSurface>& vertices) const {
    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Failed to open mesh file");
    }
    _file->read(file_stream, vertices);
    mesh_quarterchord(vertices);
}
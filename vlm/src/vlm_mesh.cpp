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

void mesh_quarterchord(f32* vertices, u64 nc, u64 ns) {
    f32* vx = vertices + 0 * (nc+1) * (ns+1);
    f32* vy = vertices + 1 * (nc+1) * (ns+1);
    f32* vz = vertices + 2 * (nc+1) * (ns+1);

    for (u64 i = 0; i < nc; i++) {
        for (u64 j = 0; j < ns+1; j++) {
            const u64 v0 = (i+0) * (ns+1) + j;
            const u64 v3 = (i+1) * (ns+1) + j;

            vx[v0] = 0.75f * vx[v0] + 0.25f * vx[v3];
            vy[v0] = 0.75f * vy[v0] + 0.25f * vy[v3];
            vz[v0] = 0.75f * vz[v0] + 0.25f * vz[v3];
        }
    }

    // Trailing edge vertices
    const u64 i = nc-1;
    for (u64 j = 0; j < ns+1; j++) {
        const u64 v0 = (i+0) * (ns+1) + j;
        const u64 v3 = (i+1) * (ns+1) + j;
        
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

    SurfDims read(std::ifstream& f, f32* vertices) const override {
        u64 ni, nj, nk, blocks;
        f32 x, y, z;
        f >> blocks;
        f >> ni >> nj >> nk;
        assert(mesh->ns == nj - 1);
        assert(mesh->nc == ni - 1);
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
        const f32 first_y =vertices[nb_vertices * 1 + 0];
        for (u64 i = 1; i < ni; i++) {
            if ( vertices[nb_vertices * 1 + i*nj] != first_y) {
                throw std::runtime_error("Mesh vertices should be ordered in chordwise direction");
            }
        }
        return {ni-1, nj-1};
    }
};

class MeshIO {
public:
    MeshIO(const std::string& format) {
        if (format == "plot3d") {
            _file = std::make_unique<Plot3DFile>();
        } else {
            throw std::runtime_error("Unsupported file format");
        }
    }

    SurfDims get_dims(const std::string& filename) const {
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

    void read(const std::string& filename, f32* vertices) const {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open mesh file");
        }
        auto [nc, ns] = _file->read(file, vertices);
        mesh_quarterchord(vertices, nc, ns);
    }

private:
    std::unique_ptr<MeshFile> _file;
};

// // TODO: DEPRECATE ALL BELOW

// // Reads the Plot3D Structured file, allocates vertex buffer and fills it with the file data
// void mesh_io_read_file_plot3d(std::ifstream& f, MeshGeom* mesh_geom) {
//     std::cout << "reading plot3d mesh" << std::endl;
//     u64 ni, nj, nk, blocks;
//     f32 x, y, z;
//     f >> blocks;
//     if (blocks != 1) {
//         throw std::runtime_error("Only single block plot3d mesh is supported");
//     }
//     f >> ni >> nj >> nk;
//     if (nk != 1) {
//         throw std::runtime_error("Only 2D plot3d mesh is supported");
//     }

//     mesh_geom->ns = nj - 1;
//     mesh_geom->nc = ni - 1;
//     mesh_geom->vertices = new f32[ni*nj*3];
//     const u64 nb_panels = mesh_geom->ns * mesh_geom->nc;
//     const u64 nb_vertices = ni * nj;

//     std::cout << "number of panels: " << nb_panels << std::endl;
//     std::cout << "ns: " << mesh_geom->ns << std::endl;
//     std::cout << "nc: " << mesh_geom->nc << std::endl;
    
//     for (u64 j = 0; j < nj; j++) {
//         for (u64 i = 0; i < ni; i++) {
//             f >> x;
//             mesh_geom->vertices[nb_vertices * 0 + nj*i + j] = x;
//         }
//     }
//     for (u64 j = 0; j < nj; j++) {
//         for (u64 i = 0; i < ni; i++) {
//             f >> y;
//             mesh_geom->vertices[nb_vertices * 1 + nj*i + j] = y;
//         }
//     }
//     for (u64 j = 0; j < nj; j++) {
//         for (u64 i = 0; i < ni; i++) {
//             f >> z;
//             mesh_geom->vertices[nb_vertices * 2 + nj*i + j] = z;
//         }
//     }

//     // TODO: this validation is not robust enough when eventually we will be using multiple bodies
//     // Validation
//     // const f32 eps = std::numeric_limits<f32>::epsilon();
//     // if (std::abs(m.v.x[0]) != eps || std::abs(m.v.y[0]) != eps) {
//     //     throw std::runtime_error("First vertex of plot3d mesh must be at origin");
//     // }
//     const f32 first_y = mesh_geom->vertices[nb_vertices * 1 + 0];
//     for (u64 i = 1; i < ni; i++) {
//         if ( mesh_geom->vertices[nb_vertices * 1 + i*nj] != first_y) {
//             throw std::runtime_error("Mesh vertices should be ordered in chordwise direction");
//         }
//     }
// }

// void vlm::mesh_io_read_file(const std::string& filename, MeshGeom* mesh_geom) {
//     const std::filesystem::path path(filename);
//     if (!std::filesystem::exists(path)) {
//         throw std::runtime_error("Mesh file not found"); // TODO: consider not using exceptions anymore
//     }

//     std::ifstream f(path);
//     try {
//         if (f.is_open()) {
//             if (path.extension() == ".x") {
//                 mesh_io_read_file_plot3d(f, mesh_geom);
//             } else {
//                 throw std::runtime_error("Only structured gridpro mesh format is supported");
//             }
//             f.close();
//         } else {
//             throw std::runtime_error("Failed to open mesh file");
//         }
//     } catch (const std::exception& e) {
//         std::cerr << e.what() << std::endl;
//     }
// }

// void vlm::mesh_quarterchord(MeshGeom* mesh_geom) {
//     const u64 nc = mesh_geom->nc;
//     const u64 ns = mesh_geom->ns;
//     f32* vx = mesh_geom->vertices + 0 * (nc+1) * (ns+1);
//     f32* vy = mesh_geom->vertices + 1 * (nc+1) * (ns+1);
//     f32* vz = mesh_geom->vertices + 2 * (nc+1) * (ns+1);

//     for (u64 i = 0; i < nc; i++) {
//         for (u64 j = 0; j < ns+1; j++) {
//             const u64 v0 = (i+0) * (ns+1) + j;
//             const u64 v3 = (i+1) * (ns+1) + j;

//             vx[v0] = 0.75f * vx[v0] + 0.25f * vx[v3];
//             vy[v0] = 0.75f * vy[v0] + 0.25f * vy[v3];
//             vz[v0] = 0.75f * vz[v0] + 0.25f * vz[v3];
//         }
//     }

//     // Trailing edge vertices
//     const u64 i = nc-1;
//     for (u64 j = 0; j < ns+1; j++) {
//         const u64 v0 = (i+0) * (ns+1) + j;
//         const u64 v3 = (i+1) * (ns+1) + j;
        
//         vx[v3] = (4.f/3.f) * vx[v3] - (1.f/3.f) * vx[v0];
//         vy[v3] = (4.f/3.f) * vy[v3] - (1.f/3.f) * vy[v0];
//         vz[v3] = (4.f/3.f) * vz[v3] - (1.f/3.f) * vz[v0];
//     }
// }

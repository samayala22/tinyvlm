#include "vlm_io.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_config.hpp"

#include "tinyconfig.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace vlm;
 
// gridpro is spanwise major
void read_gridpro_structured(std::ifstream& f, Mesh& m) {
    std::cout << "reading gridpro mesh" << std::endl;

    u32 v_nc = 0; // number of vertices chordwise
    u32 v_ns = 0; // number of vertices spanwise
    f32 x, y, z;
    f >> v_nc >> v_ns;
    m.ns = v_ns - 1;
    m.nc = v_nc - 1;

    m.alloc();
    std::cout << "number of panels: " << m.nb_panels_wing() << std::endl;

    for (u32 i = 0; i < m.nb_vertices_wing(); i++) {
        f >> x >> y >> z;
        m.v.x[i] = x;
        m.v.y[i] = y;
        m.v.z[i] = z;
    }

    for (u32 i = 0; i < m.nb_panels_wing(); i++) {
        const u32 row = i / m.ns; // row is chordwise
        const u32 ie = i + row; // index of upper left vertex of panel
        const u32 v_ns = m.ns + 1;
        m.v0.x[i] = m.v.x[ie];
        m.v0.y[i] = m.v.y[ie];
        m.v0.z[i] = m.v.z[ie];

        m.v1.x[i] = m.v.x[ie + 1];
        m.v1.y[i] = m.v.y[ie + 1];
        m.v1.z[i] = m.v.z[ie + 1];

        m.v2.x[i] = m.v.x[ie + v_ns + 1];
        m.v2.y[i] = m.v.y[ie + v_ns + 1];
        m.v2.z[i] = m.v.z[ie + v_ns + 1];

        m.v3.x[i] = m.v.x[ie + v_ns];
        m.v3.y[i] = m.v.y[ie + v_ns];
        m.v3.z[i] = m.v.z[ie + v_ns];
    }
}

// plot3d is chordwise major
void read_plot3d_structured(std::ifstream& f, Mesh& m) {
    std::cout << "reading plot3d mesh" << std::endl;
    u32 ni = 0; // number of vertices chordwise
    u32 nj = 0; // number of vertices spanwise
    u32 nk = 0;
    u32 blocks = 0;
    f32 x, y, z;
    f >> blocks;
    if (blocks != 1) {
        throw std::runtime_error("Only single block plot3d mesh is supported");
    }
    f >> ni >> nj >> nk;
    if (nk != 1) {
        throw std::runtime_error("Only 2D plot3d mesh is supported");
    }
    m.ns = nj - 1;
    m.nc = ni - 1;
    m.alloc();
    std::cout << "number of panels: " << m.nb_panels_wing() << std::endl;
    std::cout << "ns: " << m.ns << std::endl;
    std::cout << "nc: " << m.nc << std::endl;
    
    // maybe consider ni major storage
    for (u32 j = 0; j < nj; j++) {
        for (u32 i = 0; i < ni; i++) {
            f >> x;
            m.v.x[nj*i + j] = x;
        }
    }
    for (u32 j = 0; j < nj; j++) {
        for (u32 i = 0; i < ni; i++) {
            f >> y;
            m.v.y[nj*i + j] = y;
        }
    }
    for (u32 j = 0; j < nj; j++) {
        for (u32 i = 0; i < ni; i++) {
            f >> z;
            m.v.z[nj*i + j] = z;
        }
    }

    for (u32 i = 0; i < m.nb_panels_wing(); i++) {
        const u32 row = i / m.ns; // row is chordwise
        const u32 ie = i + row; // index of upper left vertex of panel
        const u32 v_ns = m.ns + 1;
        m.v0.x[i] = m.v.x[ie];
        m.v0.y[i] = m.v.y[ie];
        m.v0.z[i] = m.v.z[ie];

        m.v1.x[i] = m.v.x[ie + 1];
        m.v1.y[i] = m.v.y[ie + 1];
        m.v1.z[i] = m.v.z[ie + 1];

        m.v2.x[i] = m.v.x[ie + v_ns + 1];
        m.v2.y[i] = m.v.y[ie + v_ns + 1];
        m.v2.z[i] = m.v.z[ie + v_ns + 1];

        m.v3.x[i] = m.v.x[ie + v_ns];
        m.v3.y[i] = m.v.y[ie + v_ns];
        m.v3.z[i] = m.v.z[ie + v_ns];
    }
}

void IO::read_mesh(Mesh& mesh) {
    std::filesystem::path path(filename_mesh);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Mesh file not found");
    }

    std::ifstream f(path);
    if (f.is_open()) {
        if (path.extension() == ".dat") {
            read_gridpro_structured(f, mesh);
        } else if (path.extension() == ".xyz") {
            read_plot3d_structured(f, mesh);
        } else {
            throw std::runtime_error("Only structured gridpro mesh format is supported");
        }
        f.close();
    } else {
        throw std::runtime_error("Failed to open mesh file");
    }
}

void IO::read_config(Config &config) {
    tiny::Config conf;
    conf.read(filename_config);
    config.alphas = conf().section("solver").get_vector<f32>("alphas", {0.0f});
    config.wake_included = conf().section("solver").get<bool>("wake_included", false);
}
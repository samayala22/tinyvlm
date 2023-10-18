#include "vlm_mesh.hpp"

#include <cassert>
#include <iostream>

using namespace vlm;

Mesh::Mesh(tiny::Config& cfg) {};

void Mesh::alloc() {
    const u32 ncw = nc + nw;
    v.resize((ncw + 1) * (ns + 1));
    offsets.resize(nc * ns + 1);
    connectivity.resize(4 * nc * ns);

    colloc.resize(ncw * ns);
    normal.resize(ncw * ns);
    area.resize(ncw * ns);
}

u32 Mesh::nb_panels_wing() const { return nc * ns; };
u32 Mesh::nb_panels_total() const { return (nc+nw) * ns; };
u32 Mesh::nb_vertices_wing() const { return (nc + 1) * (ns + 1); };
u32 Mesh::nb_vertices_total() const { return (nc + nw + 1) * (ns + 1); };

f32 Mesh::chord_root() const {
    const u32 first = 0;
    const u32 last = (ns+1) * nc;
    return std::sqrt(pow<2>(v.x[first] - v.x[last]) + pow<2>(v.y[first] - v.y[last]) + pow<2>(v.z[first] - v.z[last]));
}

f32 Mesh::chord_tip() const {
    u32 first = ns;
    u32 last = (nc + 1) * (ns + 1) - 1;
    return std::sqrt(pow<2>(v.x[first] - v.x[last]) + pow<2>(v.y[first] - v.y[last]) + pow<2>(v.z[first] - v.z[last]));
}

f32 Mesh::chord_avg() const {
    return 0.5f * (chord_root() + chord_tip());
}

Vec3<f32> Mesh::get_v0(u32 i) const {
    const u32 idx = i + i / ns;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

Vec3<f32> Mesh::get_v1(u32 i) const {
    const u32 idx = i + i / ns + 1;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

Vec3<f32> Mesh::get_v2(u32 i) const {
    const u32 idx = i + i / ns + ns + 2;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

Vec3<f32> Mesh::get_v3(u32 i) const {
    const u32 idx = i + i / ns + ns + 1;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

void Mesh::update_wake(const Vec3<f32>& u_inf) {
    const f32 off_x = u_inf.x * 100.0f * chord_root();
    const f32 off_y = u_inf.y * 100.0f * chord_root();
    const f32 off_z = u_inf.z * 100.0f * chord_root();

    assert(v.x.size() == nb_vertices_total());

    const u32 v_ns = ns + 1;
    const u32 begin_trailing_edge = nb_vertices_wing()-v_ns;
    const u32 end_trailing_edge = nb_vertices_wing();
    // Add one layer of wake vertices
    // this can be parallelized (careful to false sharing tho)
    for (u32 i = begin_trailing_edge; i < end_trailing_edge; ++i) {
        v.x[i + v_ns] = v.x[i] + off_x;
        v.y[i + v_ns] = v.y[i] + off_y;
        v.z[i + v_ns] = v.z[i] + off_z;
    }
    compute_metrics_wake();
}

void Mesh::compute_connectivity() {
    // indices of the points forming the quad
    //             span(y) ->
    //  chord(x)     0--1
    //     |         |  |
    //    \/         3--2
    std::array<u32, 4> quad = {};

    // Note: with some modifications this loop can be parallelized
    for (u32 i = 0; i < nb_panels_wing(); i++) {
        // extra offset occuring from the fact that there is 1 more vertex
        // per row than surfaces
        u32 chord_idx = i / nc;
        quad[0] = i + chord_idx;
        quad[1] = quad[0] + 1;
        quad[2] = quad[1] + ns;
        quad[3] = quad[0] + ns;
        offsets[i+1] = offsets[i] + 4;
        connectivity.insert(connectivity.end(), quad.begin(), quad.end());
    }
}

void Mesh::compute_metrics_i(u32 i ) {
    Vec3<f32> v0 = get_v0(i);
    Vec3<f32> v1 = get_v1(i);
    Vec3<f32> v2 = get_v2(i);
    Vec3<f32> v3 = get_v3(i);

    // Vector v0 from p1 to p3
    const f32 v0x = v3.x - v1.x;
    const f32 v0y = v3.y - v1.y;
    const f32 v0z = v3.z - v1.z;

    // Vector v1 from p0 to p2
    const f32 v1x = v2.x - v0.x;
    const f32 v1y = v2.y - v0.y;
    const f32 v1z = v2.z - v0.z;

    normal.x[i] = (v0y*v1z - v0z*v1y);
    normal.y[i] = (v0z*v1x - v0x*v1z);
    normal.z[i] = (v0x*v1y - v0y*v1x);

    const f32 mag = std::sqrt(normal.x[i]*normal.x[i] + normal.y[i]*normal.y[i] + normal.z[i]*normal.z[i]);

    normal.x[i] /= mag;
    normal.y[i] /= mag;
    normal.z[i] /= mag;

    // 3 vectors f (P0P2), b (P0P3), e (P0P1) to compute the area:
    // area = 0.5 * (||f x b|| + ||b x e||)
    // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
    const f32 fx = v2.x - v0.x;
    const f32 fy = v2.y - v0.y;
    const f32 fz = v2.z - v0.z;

    const f32 bx = v3.x - v0.x;
    const f32 by = v3.y - v0.y;
    const f32 bz = v3.z - v0.z;

    const f32 ex = v1.x - v0.x;
    const f32 ey = v1.y - v0.y;
    const f32 ez = v1.z - v0.z;

    // s1 = f x b
    const f32 s1x = fy * bz - fz * by;
    const f32 s1y = fz * bx - fx * bz;
    const f32 s1z = fx * by - fy * bx;

    // s2 = b x e
    const f32 s2x = by * ez - bz * ey;
    const f32 s2y = bz * ex - bx * ez;
    const f32 s2z = bx * ey - by * ex;

    // area = 0.5 * (||s1|| + ||s2||)
    area[i] = 0.5f * (std::sqrt(s1x * s1x + s1y * s1y + s1z * s1z) + std::sqrt(s2x * s2x + s2y * s2y + s2z * s2z));
    
    // collocation point (center of the quad)
    colloc.x[i] = (v0.x + v1.x + v2.x + v3.x) * 0.25f;
    colloc.y[i] = (v0.y + v1.y + v2.y + v3.y) * 0.25f;
    colloc.z[i] = (v0.z + v1.z + v2.z + v3.z) * 0.25f;
}

void Mesh::compute_metrics_wing() {
    for (u32 i = 0; i < nb_panels_wing(); i++) {
        compute_metrics_i(i);
    }
}

void Mesh::compute_metrics_wake() {
    for (u32 i = nb_panels_wing(); i < nb_panels_total(); i++) {
        compute_metrics_i(i);
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
}

void Mesh::io_read(const std::string& filename) {
    std::filesystem::path path(filename);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Mesh file not found");
    }

    std::ifstream f(path);
    if (f.is_open()) {
        if (path.extension() == ".x") {
            read_plot3d_structured(f, *this);
        } else {
            throw std::runtime_error("Only structured gridpro mesh format is supported");
        }
        f.close();
    } else {
        throw std::runtime_error("Failed to open mesh file");
    }
}
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

    v0.resize(ncw * ns);
    v1.resize(ncw * ns);
    v2.resize(ncw * ns);
    v3.resize(ncw * ns);

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

void Mesh::update_wake(const Vec3& u_inf) {
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

    // note: this only works for a single row of wake panels
    for (u32 i = nb_panels_wing(); i < nb_panels_total(); i++) {
        const u32 row = i / ns; // row is chordwise
        const u32 ie = i + row; // index of upper left vertex of panel
        // note: in reality v0 and v1 dont change (only v2 and v3 do)
        v0.x[i] = v.x[ie];
        v0.y[i] = v.y[ie];
        v0.z[i] = v.z[ie];

        v1.x[i] = v.x[ie + 1];
        v1.y[i] = v.y[ie + 1];
        v1.z[i] = v.z[ie + 1];

        v2.x[i] = v.x[ie + v_ns + 1];
        v2.y[i] = v.y[ie + v_ns + 1];
        v2.z[i] = v.z[ie + v_ns + 1];

        v3.x[i] = v.x[ie + v_ns];
        v3.y[i] = v.y[ie + v_ns];
        v3.z[i] = v.z[ie + v_ns];
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
    // Vector v0 from p1 to p3
    const f32 v0x = v3.x[i] - v1.x[i];
    const f32 v0y = v3.y[i] - v1.y[i];
    const f32 v0z = v3.z[i] - v1.z[i];

    // Vector v1 from p0 to p2
    const f32 v1x = v2.x[i] - v0.x[i];
    const f32 v1y = v2.y[i] - v0.y[i];
    const f32 v1z = v2.z[i] - v0.z[i];

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
    const f32 fx = v2.x[i] - v0.x[i];
    const f32 fy = v2.y[i] - v0.y[i];
    const f32 fz = v2.z[i] - v0.z[i];

    const f32 bx = v3.x[i] - v0.x[i];
    const f32 by = v3.y[i] - v0.y[i];
    const f32 bz = v3.z[i] - v0.z[i];

    const f32 ex = v1.x[i] - v0.x[i];
    const f32 ey = v1.y[i] - v0.y[i];
    const f32 ez = v1.z[i] - v0.z[i];

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
    colloc.x[i] = (v0.x[i] + v1.x[i] + v2.x[i] + v3.x[i]) * 0.25f;
    colloc.y[i] = (v0.y[i] + v1.y[i] + v2.y[i] + v3.y[i]) * 0.25f;
    colloc.z[i] = (v0.z[i] + v1.z[i] + v2.z[i] + v3.z[i]) * 0.25f;
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
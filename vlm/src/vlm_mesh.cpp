#include "vlm_mesh.hpp"
#include "linalg.h"
#include "vlm_types.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <memory>

using namespace vlm;

constexpr f32 EPS = std::numeric_limits<f32>::epsilon();

Mesh::Mesh(const tiny::Config& cfg)
{
    s_ref = cfg().section("solver").get<f32>("s_ref", 0.0f);
    c_ref = cfg().section("solver").get<f32>("c_ref", 0.0f);
    const std::vector<f32> ref_pt_vec = cfg().section("solver").get_vector<f32>("ref_pt", {0.25f, 0.0f, 0.0f});
    // TODO: maybe need an assert here in case vector size != 3
    ref_pt.x = ref_pt_vec[0];
    ref_pt.y = ref_pt_vec[1];
    ref_pt.z = ref_pt_vec[2];
    io_read(cfg().section("files").get<std::string>("mesh"));
    init();
}

Mesh::Mesh(
    const std::string& filename,
    const f32 s_ref_ = 0.0f,
    const f32 c_ref_ = 0.0f,
    const linalg::alias::float3& ref_pt_ = {0.25f, 0.0f, 0.0f}
    ) {
    s_ref = s_ref_;
    c_ref = c_ref_;
    ref_pt = ref_pt_;    
    io_read(filename);
    init();
}

std::unique_ptr<Mesh> vlm::create_mesh(const std::string filename) {
    return std::make_unique<Mesh>(filename);
}

void Mesh::init() {
    if (c_ref == 0.0f) c_ref = chord_mean(0, ns+1);
    if (s_ref == 0.0f) s_ref = panels_area_xy(0,0, nc, ns);
    compute_connectivity();
    compute_metrics_wing();
}

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

/// @brief Computes the mean chord of a set of panels
/// @details
/// Mean Aerodynamic Chord = \frac{2}{S} \int_{0}^{b/2} c(y)^{2} dy
/// Integration using the Trapezoidal Rule
/// Validated against empirical formulas for tapered wings
/// @param j first panel index chordwise
/// @param n number of panels spanwise
/// @return mean chord of the set of panels
f32 Mesh::chord_mean(const u32 j, const u32 n) const {
    assert(j + n <= ns+1); // spanwise range
    assert(n > 1); // minimum 2 chords
    f32 mac = 0.0f;
    // loop over panel chordwise sections in spanwise direction
    // Note: can be done optimally with vertical fused simd
    for (u32 v = 0; v < n - 1; v++) {
        const f32 c0 = chord_length(j + v);
        const f32 c1 = chord_length(j + v + 1);
        mac += 0.5f * (c0 * c0 + c1 * c1) * panel_width_y(0, j + v);
    }
    // Since we divide by half the total wing area (both sides) we dont need to multiply by 2
    return mac / panels_area_xy(0, j, nc, n-1);
}

/// @brief Computes the total area of a 2D set of panels
/// @param i first panel index chordwise
/// @param j first panel panel index spanwise
/// @param m number of panels chordwise
/// @param n number of panels spanwise
/// @return sum of the areas of the panels
f32 Mesh::panels_area(const u32 i, const u32 j, const u32 m, const u32 n) const {
    assert(i + m <= nc);
    assert(j + n <= ns);
    
    const u32 ld = ns;
    const f32* areas = &area[j + i * ld];

    f32 total_area = 0.0f;
    for (u32 u = 0; u < m; u++) {
        for (u32 v = 0; v < n; v++) {
            total_area += areas[v + u * ld];
        }
    }
    return total_area;
}

/// @brief Computes the total area of a 2D set of panels projected on the xy plane
/// @param i first panel index chordwise
/// @param j first panel panel index spanwise
/// @param m number of panels chordwise
/// @param n number of panels spanwise
/// @return sum of the areas of the panels projected on the xy plane
f32 Mesh::panels_area_xy(const u32 i, const u32 j, const u32 m, const u32 n) const {
    assert(i + m <= nc);
    assert(j + n <= ns);
    
    // Area of a quad:
    // 0.5 * || d1 x d2 || where d1 and d2 are the diagonals of the quad
    f32 total_area = 0.0f;
    // Note: this is highly inefficient as vertex coordinates should be loaded with simd
    for (u32 u = 0; u < m; u++) {
        for (u32 v = 0; v < n; v++) {
            const u32 idx = j + v + (i+u) * ns;
            const linalg::alias::float3 d1 = get_v0(idx) - get_v2(idx);
            const linalg::alias::float3 d2 = get_v1(idx) - get_v3(idx);
            const linalg::alias::float3 cross = linalg::cross(d1, d2);
            
            total_area += 0.5f * std::abs(cross.z);
            // std::cout << "area xy: " << 0.5f * std::abs(cross.z()) << "\n";
            // std::cout << "area: " << area[idx] << "\n";
        }
    }
    return total_area;
}

/// @brief Computes the width of a single panel in pure y direction
/// @param i panel index chordwise
/// @param j panel index spanwise
f32 Mesh::panel_width_y(const u32 i, const u32 j) const {
    // Since chordwise segments are always parallel, we can simply measure the width of the first panel
    assert(i < nc);
    assert(j < ns);
    const u32 ld = ns + 1;
    return v.y[j + 1 + i * ld] - v.y[j + i * ld];
}

/// @brief Computes the chord length of a chordwise segment
/// @details Since the mesh follows the camber line, the chord length is computed
/// as the distance between the first and last vertex of a chordwise segment
/// @param j chordwise segment index
f32 Mesh::chord_length(const u32 j) const {
    assert(j <= ns); // spanwise vertex range
    const f32 dx = v.x[j + nc * (ns+1)] - v.x[j];
    const f32 dy = 0.0f; // chordwise segments are parallel to the x axis
    const f32 dz = v.z[j + nc * (ns+1)] - v.z[j];

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

linalg::alias::float3 Mesh::get_v0(u32 i) const {
    const u32 idx = i + i / ns;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

linalg::alias::float3 Mesh::get_v1(u32 i) const {
    const u32 idx = i + i / ns + 1;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

linalg::alias::float3 Mesh::get_v2(u32 i) const {
    const u32 idx = i + i / ns + ns + 2;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

linalg::alias::float3 Mesh::get_v3(u32 i) const {
    const u32 idx = i + i / ns + ns + 1;
    return {v.x[idx], v.y[idx], v.z[idx]};
}

void Mesh::update_wake(const linalg::alias::float3& freestream) {
    const f32 chord_root = chord_length(0); 
    const f32 off_x = freestream.x * 100.0f * chord_root;
    const f32 off_y = freestream.y * 100.0f * chord_root;
    const f32 off_z = freestream.z * 100.0f * chord_root;

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

// https://publications.polymtl.ca/2555/1/2017_MatthieuParenteau.pdf (Eq 3.4 p21)
void Mesh::correction_high_aoa(f32 alpha_rad) {
    const f32 factor = 0.5f * alpha_rad / (std::sin(alpha_rad) + EPS); // correction factor
    // Note: this can be vectorized and parallelized
    for (u32 i = 0; i < nb_panels_total(); i++) {
        // "chord vector" from center of leading line (v0-v1) to trailing line (v3-v2)
        const linalg::alias::float3 chord_vec = 0.5f * (get_v2(i) + get_v3(i) - get_v0(i) - get_v1(i));
        const linalg::alias::float3 colloc_pt = 0.5f * (get_v0(i) + get_v1(i)) + factor * chord_vec;
        // collocation calculated as a translation of the center of leading line center
        colloc.x[i] = colloc_pt.x;
        colloc.y[i] = colloc_pt.y;
        colloc.z[i] = colloc_pt.z;
    }
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
    const linalg::alias::float3 v0 = get_v0(i);
    const linalg::alias::float3 v1 = get_v1(i);
    const linalg::alias::float3 v2 = get_v2(i);
    const linalg::alias::float3 v3 = get_v3(i);

    // Vector v0 from p1 to p3
    const linalg::alias::float3 vec_0 = v3 - v1;

    // Vector v1 from p0 to p2
    const linalg::alias::float3 vec_1 = v2 - v0;

    // Normal = v0 x v1
    const linalg::alias::float3 normal_vec = linalg::normalize(linalg::cross(vec_0, vec_1));
    normal.x[i] = normal_vec.x;
    normal.y[i] = normal_vec.y;
    normal.z[i] = normal_vec.z;

    // 3 vectors f (P0P3), b (P0P2), e (P0P1) to compute the area:
    // area = 0.5 * (||f x b|| + ||b x e||)
    // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
    const linalg::alias::float3 vec_f = v3 - v0;
    const linalg::alias::float3 vec_b = v2 - v0;
    const linalg::alias::float3 vec_e = v1 - v0;

    area[i] = 0.5f * (linalg::length(linalg::cross(vec_f, vec_b)) + linalg::length(linalg::cross(vec_b, vec_e)));
    
    // collocation point (center of the quad)
    const linalg::alias::float3 colloc_pt = 0.25f * (v0 + v1 + v2 + v3);
    
    colloc.x[i] = colloc_pt.x;
    colloc.y[i] = colloc_pt.y;
    colloc.z[i] = colloc_pt.z;
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
void Mesh::io_read_plot3d_structured(std::ifstream& f) {
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

    ns = nj - 1;
    nc = ni - 1;
    alloc();

    std::cout << "number of panels: " << nb_panels_wing() << std::endl;
    std::cout << "ns: " << ns << std::endl;
    std::cout << "nc: " << nc << std::endl;
    
    for (u32 j = 0; j < nj; j++) {
        for (u32 i = 0; i < ni; i++) {
            f >> x;
            v.x[nj*i + j] = x;
        }
    }
    for (u32 j = 0; j < nj; j++) {
        for (u32 i = 0; i < ni; i++) {
            f >> y;
            v.y[nj*i + j] = y;
        }
    }
    for (u32 j = 0; j < nj; j++) {
        for (u32 i = 0; i < ni; i++) {
            f >> z;
            v.z[nj*i + j] = z;
        }
    }

    // TODO: this validation is not robust enough when eventually we will be using multiple bodies
    // Validation
    // const f32 eps = std::numeric_limits<f32>::epsilon();
    // if (std::abs(m.v.x[0]) != eps || std::abs(m.v.y[0]) != eps) {
    //     throw std::runtime_error("First vertex of plot3d mesh must be at origin");
    // }
    f32 first_y = v.y[0];
    for (u32 i = 1; i < ni; i++) {
        if (v.y[i * nj] != first_y) {
            throw std::runtime_error("Mesh vertices should be ordered in chordwise direction");
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
            io_read_plot3d_structured(f);
        } else {
            throw std::runtime_error("Only structured gridpro mesh format is supported");
        }
        f.close();
    } else {
        throw std::runtime_error("Failed to open mesh file");
    }
}
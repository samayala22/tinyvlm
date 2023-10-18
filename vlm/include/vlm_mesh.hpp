#pragma once

#include "vlm_types.hpp"

namespace vlm {

// === STRUCTURED MESH ===
// nc : number of panels chordwise
// ns : number of panels spanwise
// nw : number of panels in chordwise wake
// ncw : nc + nw
struct Mesh {
    // Unstructured members (for exporting results)
    // ---------------------
    // All vertices stored in single SoA for result exporting
    // (stored in span major order)
    SoA_3D_t<f32> v; // size (ncw+1)*(ns+1)
    // Offsets for indexing in connectivity array for each panel
    std::vector<u32> offsets = {}; // size nc*ns + 1
    // Panel-vertex connectivity
    // vertices indices of a panel: connectivity[offsets[i]] to connectivity[offsets[i+1]]
    std::vector<u32> connectivity = {}; // size 4*(nc*ns)

    // Panels metrics (stored span major order)
    // ---------------------
    // Collocation points of panels
    SoA_3D_t<f32> colloc; // size ncw*ns
    // Normals of panels
    SoA_3D_t<f32> normal; // size ncw*ns
    // Area of panels
    std::vector<f32> area = {}; // size ncw*ns

    // Structured dimensions
    // ---------------------
    u32 nc = 1; // number of panels chordwise
    u32 ns = 1; // number of panels spanwise
    const u32 nw = 1; // number of panels in chordwise wake

    void alloc(); // allocate memory for all arrays
    void update_wake(const Vec3<f32>& u_inf); // update wake vertices
    void correction_high_aoa(f32 alpha_rad); // correct collocation points for high aoa
    void compute_connectivity(); // fill offsets, connectivity
    void compute_metrics_wing(); // fill colloc, normal, area
    void compute_metrics_wake();
    void compute_metrics_i(u32 i);
    u32 nb_panels_wing() const;
    u32 nb_panels_total() const;
    u32 nb_vertices_wing() const;
    u32 nb_vertices_total() const;
    f32 chord_root() const;
    f32 chord_tip() const;
    f32 chord_avg() const;
    // i panel vertices coordinates
    Vec3<f32> get_v0(u32 i) const; // upper left
    Vec3<f32> get_v1(u32 i) const; // upper right
    Vec3<f32> get_v2(u32 i) const; // lower right
    Vec3<f32> get_v3(u32 i) const; // lower left

    void io_read(const std::string& filename);
    Mesh() = default;
    Mesh(tiny::Config& cfg);
};

} // namespace vlm
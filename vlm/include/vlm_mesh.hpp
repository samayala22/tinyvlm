#pragma once

#include "vlm_types.hpp"
#include "vlm_allocator.hpp"

#include "tinyfwd.hpp"
#include "tinyview.hpp"

namespace vlm {

// C interface
struct Mesh2 {
    f32* vertices = nullptr; // (nc+nw+1)*(ns+1)*3
    // TODO: evaluate if we really need those values or we can just compute on the fly in the kernels
    f32* normals = nullptr; // nc*ns*3
    f32* colloc = nullptr; // nc*ns*3
    f32* area = nullptr; // nc*ns

    u64 nc = 0; // number of wing panels chordwise
    u64 ns = 0; // number of wing panels spanwise
    u64 nw = 0; // number of wake panels chordwise
    u64 nwa = 0; // number of wake panels active chordwise

    f32 s_ref = 0.0f; // reference area
    f32 c_ref = 0.0f; // reference chord
    f32 frame[16] = {}; // 4x4 homogeneous matrix in col major order

    char* name = nullptr; // mesh name
    bool lifting = true;
};

// Contains only the raw geometry 
struct MeshGeom {
    f32* vertices = nullptr;
    u64 nc = 0; // number of wing panels chordwise
    u64 ns = 0;
};

void mesh_read_file(const std::string& filename, MeshGeom* mesh_geom);
void mesh_alloc(const malloc_f malloc, Mesh2* mesh, u64 nc, u64 ns, u64 nw, u64 nwa);
void mesh_free(const free_f free, Mesh2* mesh);
void mesh_copy(const memcpy_f memcpy, Mesh2* dst, const Mesh2* src);

// === STRUCTURED MESH ===
// nc : number of panels chordwise
// ns : number of panels spanwise
// nw : number of panels in chordwise wake
// ncw : nc + nw
class Mesh {
    public:
    // Unstructured members (for exporting results)
    // ---------------------
    // All vertices stored in single SoA for result exporting
    // (stored in span major order)
    SoA_3D_t<f32> v; // size (ncw+1)*(ns+1)
    // Offsets for indexing in connectivity array for each panel
    std::vector<u64> offsets = {}; // size nc*ns + 1
    // Panel-vertex connectivity
    // vertices indices of a panel: connectivity[offsets[i]] to connectivity[offsets[i+1]]
    std::vector<u64> connectivity = {}; // size 4*(nc*ns)

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
    u64 nc = 1; // number of panels chordwise
    u64 ns = 1; // number of panels spanwise
    u64 nw; // number of wake panels chordwise (max capacity)
    u64 current_nw = 0; // current number of built wake panels (temporary)

    // Analytical quanities when provided
    // ---------------------
    f32 s_ref = 0.0f; // reference area (of wing)
    f32 c_ref = 0.0f; // reference chord (of wing)

    linalg::alias::float4x4 frame = linalg::identity; // transformation matrix
    linalg::alias::float3 ref_pt = {0.25f, 0.0f, 0.0f}; // TODO: deprecate

    void update_wake(const linalg::alias::float3& u_inf); // update wake vertices
    void correction_high_aoa(f32 alpha_rad); // correct collocation points for high aoa
    void create_vortex_panels(); // create true vortex panels
    void compute_connectivity(); // fill offsets, connectivity
    void compute_metrics_wing(); // fill colloc, normal, area
    void compute_metrics_wake();
    void compute_metrics_i(u64 i);
    void move(const linalg::alias::float4x4& transform, const SoA_3D_t<f32>& origin_pos);
    void resize_wake(const u64 nw);
 
    u64 nb_panels_wing() const;
    u64 nb_panels_total() const;
    u64 nb_vertices_wing() const;
    u64 nb_vertices_total() const;
    f32 panels_area(const u64 i, const u64 j, const u64 m, const u64 n) const;
    f32 panels_area_xy(const u64 i, const u64 j, const u64 m, const u64 n) const;
    f32 panel_width_y(const u64 i, const u64 j) const;
    f32 panel_length(const u64 i, const u64 j) const;
    f32 strip_width(const u64 j) const;
    f32 chord_length(const u64 j) const; // vertex idx
    f32 chord_mean(const u64 j, const u64 n) const;

    // i panel vertices coordinates
    linalg::alias::float3 get_v0(u64 i) const; // upper left
    linalg::alias::float3 get_v1(u64 i) const; // upper right
    linalg::alias::float3 get_v2(u64 i) const; // lower right
    linalg::alias::float3 get_v3(u64 i) const; // lower left

    void io_read(const std::string& filename);
    Mesh(const tiny::Config& cfg);
    Mesh(
        const std::string& filename,
        const u64 nw,
        const f32 s_ref_,
        const f32 c_ref_,
        const linalg::alias::float3& ref_pt_ // todo: deprecate
    );
    
    private:
    void alloc(); // allocate memory for all buffers
    void init(); // called at the end of constructor
    void io_read_plot3d_structured(std::ifstream& f);
};

// todo, update this to mirror the constructor
std::unique_ptr<Mesh> create_mesh(const std::string& filename, const u64 nw=1);

} // namespace vlm
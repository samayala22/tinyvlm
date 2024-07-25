#pragma once

#include <iosfwd>

#include "vlm_types.hpp"
#include "vlm_memory.hpp"

#include "tinyfwd.hpp"

// #define PTR_MESHGEOM_V(m, i,j,k) (m->vertices + (j) + (i) * (m->ns+1) + (k) * (m->nc+1) * (m->ns+1))
// #define PTR_MESH_V(m, i,j,k) (m->vertices + (j) + (i) * (m->ns+1) + (k) * (m->nc+m->nw+1) * (m->ns+1))
// #define PTR_MESH_C(m, i,j,k) (m->colloc + (j) + (i) * (m->ns) + (k) * (m->nc) * (m->ns))
// #define PTR_MESH_N(m, i,j,k) (m->normals + (j) + (i) * (m->ns) + (k) * (m->nc) * (m->ns))

namespace vlm {

constexpr inline f32* PTR_P(f32* ptr, u64 ni, u64 nj, u64 i, u64 j, u64 k) {
    return ptr + j + i*nj + k*ni*nj;
}
constexpr inline f32* PTR_V(f32* ptr, u64 ni, u64 nj, u64 i, u64 j, u64 k) {
    return ptr + j + i*(nj+1) + k*(ni+1)*(nj+1);
}

struct MeshParams {
    u64 nc = 0; // nb of panels chordwise
    u64 ns = 0; // nb of panels spanwise
    u64 nw = 0; // nb of wake panels chordwise
    u64 nwa = 0; // nb of acive wake panels chordwise
    u64 off_wing_p = 0; // wing panel offset for buffers
    u64 off_wing_v = 0; // wing vertex offset for buffers
    u64 off_wake_p = 0; // wake panel offset for buffers
    u64 off_wake_v = 0; // wake vertex offset for buffers
    bool wake = true; // whether the surface produces wake

    inline u64 nb_panels_wing() const {return nc * ns;}
    inline u64 nb_panels_wake() const {return nw * ns;}
    inline u64 nb_vertices_wing() const {return (nc+1) * (ns+1);}
    inline u64 nb_vertices_wake() const {return (nw+1) * (ns+1);}
};

// TODO: move functions into cpp file
struct Mesh2 {
    Buffer<f32, MemoryLocation::HostDevice> verts_wing_init; // (nc+1)*(ns+1)*3
    Buffer<f32, MemoryLocation::HostDevice> verts_wing; // (nc+1)*(ns+1)*3
    Buffer<f32, MemoryLocation::HostDevice> verts_wake; // (nw+1)*(ns+1)*3
    Buffer<f32, MemoryLocation::Device> normals; // nc*ns*3
    Buffer<f32, MemoryLocation::Device> colloc; // nc*ns*3
    Buffer<f32, MemoryLocation::Device> area; // nc*ns

    std::vector<MeshParams> params;

    f32 frame[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    }; // Col major order (TODO: move this to kinematic tracker?)

    Mesh2(const Memory& memory) : 
    verts_wing_init(memory), verts_wing(memory), verts_wake(memory),
    normals(memory), colloc(memory), area(memory) {};
    
    void alloc_wing() {
        verts_wing_init.alloc(total_nb_vertices_wing()*4);
        verts_wing.alloc(total_nb_vertices_wing()*4);
        normals.alloc(total_nb_panels_wing()*3);
        colloc.alloc(total_nb_panels_wing()*3);
        area.alloc(total_nb_panels_wing());

        // Sete last row to 1 for homogeneous coordinates
        verts_wing_init.memory.fill_f32(MemoryLocation::Host, verts_wing_init.h_ptr() + total_nb_vertices_wing()*3, 1.f, total_nb_vertices_wing());
    }

    void alloc_wake() {
        verts_wake.alloc(total_nb_vertices_wake()*4);
        verts_wake.memory.fill_f32(MemoryLocation::Device, verts_wake.d_ptr() + total_nb_vertices_wake()*3, 1.f, total_nb_vertices_wake()); // not necessary but kept for consistency
    }
    
    u64 total_nb_panels_wing() { return params.back().off_wing_p + params.back().nb_panels_wing(); }
    u64 total_nb_panels_wake() { return params.back().off_wake_p + params.back().nb_panels_wake(); }
    u64 total_nb_vertices_wing() { return params.back().off_wing_v + params.back().nb_vertices_wing(); }
    u64 total_nb_vertices_wake() { return params.back().off_wake_v + params.back().nb_vertices_wake(); }
};

using SurfDims = std::pair<u64, u64>; // nc, ns

class MeshFile {
public:
    virtual ~MeshFile() = default;
    virtual SurfDims get_dims(std::ifstream& file) const = 0;
    virtual SurfDims read(std::ifstream& file, f32* vertices) const = 0;
};

class MeshIO {
public:
    MeshIO(const std::string& format);
    SurfDims get_dims(const std::string& filename) const;
    void read(const std::string& filename, f32* vertices) const;
    // void write(const std::string& filename, const Mesh* mesh) const;
private:
    std::unique_ptr<MeshFile> _file;
};

} // namespace vlm

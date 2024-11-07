#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "vlm_types.hpp"
#include "vlm_memory.hpp"

#include "tinyfwd.hpp"

namespace vlm {

// TODO: move functions into cpp file
class Mesh {
    public:
    Buffer<f32, Location::HostDevice, MultiSurface> verts_wing_init; // (nc+1)*(ns+1)*3
    Buffer<f32, Location::HostDevice, MultiSurface> verts_wing; // (nc+1)*(ns+1)*3
    Buffer<f32, Location::HostDevice, MultiSurface> verts_wake; // (nw+1)*(ns+1)*3
    Buffer<f32, Location::HostDevice, MultiSurface> area; // nc*ns

    f32 frame[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    }; // Col major order (TODO: move this to kinematic tracker?)

    Mesh(const Memory& memory) : 
    verts_wing_init(memory), verts_wing(memory), verts_wake(memory),
    area(memory) {};
    
    void alloc_wing(const std::vector<SurfaceDims>& wing_panels, const std::vector<SurfaceDims>& wing_vertices) {
        verts_wing_init.alloc(MultiSurface{wing_vertices, 4});
        verts_wing.alloc(MultiSurface{wing_vertices, 4});
        area.alloc(MultiSurface{wing_panels, 1});

        // Sete last row to 1 for homogeneous coordinatesverts_wing_init.h_view().layout
        verts_wing_init.memory.fill(Location::Host, verts_wing_init.h_view().ptr + verts_wing_init.h_view().layout(0,3), 1, 1.f, verts_wing_init.h_view().layout.stride());
    }

    void alloc_wake(const std::vector<SurfaceDims>& wake_vertices) {
        verts_wake.alloc(MultiSurface{wake_vertices, 4});
        verts_wake.memory.fill(Location::Device, verts_wake.d_view().ptr + verts_wake.d_view().layout(0,3), 1, 1.f,  verts_wake.d_view().layout.stride());
    }
};

template<i32 N>
class Assembly {
    using Shape = std::array<i64, N>;
    private:
        std::vector<Shape> m_shapes;
        std::vector<i64> m_offsets;

        i64 shape_size(const Shape& shape) const {
            i64 size = 1;
            for (i64 i = 0; i < N; i++) size *= shape[i];
            return size;
        }
    public:
        Shape shape(i64 i) const { assert(i < N); return m_shapes[i]; }
        i64 begin(i64 i) const { assert(i < N); return m_offsets[i]; }
        i64 end(i64 i) const { assert(i < N); return begin(i) + shape_size(shape(i)); }
        i64 size() const { return m_shapes.size(); }
        void add(const Shape& shape) {
            m_shapes.push_back(shape);
            if (m_offsets.size() == 0) {
                m_offsets.push_back(0);
            } else {
                m_offsets.push_back(m_offsets.back() + shape_size(shape));
            }
        }
        void clear() {
            m_shapes.clear();
            m_offsets.clear();
        }
};

using Assembly3D = Assembly<3>;
using Assembly2D = Assembly<2>;
using Assembly1D = Assembly<1>;

using SurfDims = std::pair<i64, i64>; // nc, ns

class MeshFile {
public:
    virtual ~MeshFile() = default;
    virtual SurfDims get_dims(std::ifstream& file) const = 0;
    virtual void read(std::ifstream& file, View<f32, SingleSurface>& vertices) const = 0;
};

class MeshIO {
public:
    MeshIO(const std::string& format);
    SurfDims get_dims(const std::string& filename) const;
    void read(const std::string& filename, View<f32, SingleSurface>& vertices) const;

private:
    std::unique_ptr<MeshFile> _file;
};

} // namespace vlm

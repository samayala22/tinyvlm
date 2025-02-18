#include <vector>
#include <string>
#include <functional> // std::function

#include "tinycombination.hpp"
#include "tinyad.hpp"
#include "tinypbar.hpp"

#include "vlm.hpp"
#include "vlm_types.hpp"
#include "vlm_kinematics.hpp"
#include "vlm_utils.hpp"

using namespace vlm;
using namespace linalg::ostream_overloads;

class HBVLM final: public Simulation {
    public:
        HBVLM(const std::string& backend_name, const Assembly& assembly, const i32 harmonics);
        ~HBVLM() = default;
        void run(f32 t_final);

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Host> colloc_h{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        MultiTensor2D<Location::Device> areas_d{backend->memory.get()};

        MultiTensor3D<Location::Device> verts_wing_pos{backend->memory.get()}; // (nc+1)*(ns+1)*3

        // Data
        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor2D<Location::Device> rhs{backend->memory.get()}; // (ns*nc) x harmonics
        MultiTensor3D<Location::Device> gamma_wing{backend->memory.get()}; // ns x nc x harmonics
        MultiTensor3D<Location::Device> aero_forces{backend->memory.get()}; // ns*nc*3
        
        MultiTensor3D<Location::Device> velocities{backend->memory.get()}; // ns*nc*3
        MultiTensor3D<Location::Host> velocities_h{backend->memory.get()}; // ns*nc*3

        MultiTensor2D<Location::Host> transforms_h{backend->memory.get()};
        MultiTensor2D<Location::Device> transforms{backend->memory.get()};
        
        std::unique_ptr<LU> solver;

        std::vector<i32> condition0;
    private:
        void alloc_buffers(const i32 harmonics);
};

// TODO: move this somewhere else
inline i64 total_panels(const MultiDim<2>& assembly_wing) {
    i64 total = 0;
    for (const auto& wing : assembly_wing) {
        total += wing[0] * wing[1];
    }
    return total;
}

void HBVLM::alloc_buffers(const i32 harmonics) {
    const i64 n = total_panels(assembly_wings);
    // Mesh
    MultiDim<3> gamma;
    MultiDim<3> panels_3D;  
    MultiDim<2> panels_2D;
    MultiDim<3> verts_wing_3D;
    MultiDim<2> transforms_2D;
    for (const auto& [ns, nc] : assembly_wings) {
        gamma.push_back({ns, nc, harmonics});
        panels_3D.push_back({ns, nc, 3});
        panels_2D.push_back({ns, nc});
        verts_wing_3D.push_back({ns+1, nc+1, 4});
        transforms_2D.push_back({4, 4});
    }
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    colloc_h.init(panels_3D);
    areas_d.init(panels_2D);

    // Data
    lhs.init({n, n});
    rhs.init({n, harmonics});
    gamma_wing.init(gamma);
    velocities.init(panels_3D);
    velocities_h.init(panels_3D);
    aero_forces.init(panels_3D);
    transforms_h.init(transforms_2D);
    transforms.init(transforms_2D);
    solver->init(lhs.view());

    condition0.resize(assembly_wings.size()*assembly_wings.size());
}

int main() {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_10x5.x"};
    const std::vector<std::string> backends = {"cpu"};

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 5.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 k = 0.5; // reduced frequency
    const f32 omega = k * u_inf / b;
    const f32 t_final = cycles * 2.0f * PI_f / omega;
    const i32 harmonics = 3;

    KinematicsTree kinematics_tree;

    // Periodic pitching
    const f32 amplitude = 3.f; // amplitude in degrees
    auto fs = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    });
    auto pitch = kinematics_tree.add([=](const fwd::Float& t) {
        return rotation_matrix<fwd::Float>({0.25f, 0.0f, 0.0f},{0.0f, 1.0f, 0.0f}, to_radians(amplitude) * fwd::sin(omega * t));
    })->after(fs);

    for (const auto& [mesh_name, backend_name] : solvers) {
        Assembly assembly(fs);
        assembly.add(mesh_name, pitch);
        // HBVLM simulation{backend_name, assembly, harmonics};
        // simulation.run(t_final);
    }
    return 0;
}
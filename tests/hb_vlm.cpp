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
        void run(f32 t_final, f32 omega);

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Host> colloc_h{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        MultiTensor2D<Location::Device> areas_d{backend->memory.get()};

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
        Assembly m_assembly;
        i32 m_harmonics;
    private:
        void alloc_buffers();
};

HBVLM::HBVLM(
    const std::string& backend_name,
    const Assembly& assembly,
    const i32 harmonics) : 
    Simulation(backend_name, assembly.mesh_filenames(), false),
    m_assembly(assembly),
    m_harmonics(harmonics)
{
    solver = backend->create_lu_solver();
    alloc_buffers();
}

// TODO: move this somewhere else
inline i64 total_panels(const MultiDim<2>& assembly_wing) {
    i64 total = 0;
    for (const auto& wing : assembly_wing) {
        total += wing[0] * wing[1];
    }
    return total;
}

void HBVLM::alloc_buffers() {
    const i64 n = total_panels(assembly_wings);
    // Mesh
    MultiDim<3> gamma;
    MultiDim<3> panels_3D;  
    MultiDim<2> panels_2D;
    MultiDim<3> verts_wing_3D;
    MultiDim<2> transforms_2D;
    for (const auto& [ns, nc] : assembly_wings) {
        gamma.push_back({ns, nc, 2*m_harmonics+1});
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
    rhs.init({n, 2*m_harmonics+1});
    gamma_wing.init(gamma);
    velocities.init(panels_3D);
    velocities_h.init(panels_3D);
    aero_forces.init(panels_3D);
    transforms_h.init(transforms_2D);
    transforms.init(transforms_2D);
    solver->init(lhs.view());

    condition0.resize(assembly_wings.size()*assembly_wings.size());
}

void HBVLM::run(f32 t_start, f32 omega) {
    const f32 period = 2.0f * PI_f / omega;
    const f32 rho = 1.0f; // TODO: take this as input

    for (const auto& [kinematics, transform_h, transform_d] : zip(m_assembly.surface_kinematics(), transforms_h.views(), transforms.views())) {
        auto transform = kinematics->transform(0.0f);
        transform.store(transform_h.ptr(), transform_h.stride(1));
        transform_h.to(transform_d);
    }
    backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);

    // 1.  Compute the fixed time step
    const auto& verts_first_wing = verts_wing_init_h.views()[0];
    const f32 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    const f32 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 1);
    const f32 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 2);
    const f32 last_panel_chord = std::sqrt(dx*dx + dy*dy + dz*dz);
    const f32 dt = last_panel_chord / linalg::length(m_assembly.kinematics()->linear_velocity(0.0f, {0.f, 0.f, 0.f}));
    const i64 max_t_steps = static_cast<i64>((t_start+period+dt) / dt);

    // 2. Allocate the wake geometry
    {
        MultiDim<2> wake_panels_2D;
        MultiDim<3> verts_wake_3D;
        for (const auto& [ns, nc] : assembly_wings) {
            wake_panels_2D.push_back({ns, max_t_steps-1});
            verts_wake_3D.push_back({ns+1, max_t_steps, 4});
        }
        verts_wake.init(verts_wake_3D);
    }

    // 3. Precompute constant values for the transient simulation
    lhs.view().fill(0.f);
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0 , 0);
    solver->factorize(lhs.view());

    // Assemble for 2N + 1 solutions
    for (i32 s = 0; s < 2*m_harmonics+1; s++) {
        // Reset wing position
        for (const auto& [kinematics, transform_h, transform_d] : zip(m_assembly.surface_kinematics(), transforms_h.views(), transforms.views())) {
            auto transform = kinematics->transform(0.0f);
            transform.store(transform_h.ptr(), transform_h.stride(1));
            transform_h.to(transform_d);
        }
        backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());

        const f32 t_final = t_start + period * (f32)s / (f32)(2*m_harmonics+1);
        const i64 t_steps = static_cast<i64>((t_final+dt) / dt);
        std::printf("t_final: %f\n", t_final);
        for (i32 i = 0; i < t_steps-1; i++) { // minus one because at the end the displace wing is at t=t+dt
            const f32 t = (f32)i * dt;
            
            backend->wake_shed(verts_wing.views(), verts_wake.views(), i);
 
            // parallel for
            for (i64 m = 0; m < assembly_wings.size(); m++) {
                auto transform = m_assembly.surface_kinematics()[m]->transform(t+dt);
                transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
                transforms_h.views()[m].to(transforms.views()[m]);
            }

            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
        }
        backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
        
        // parallel for
        for (i64 m = 0; m < assembly_wings.size(); m++) {
            const auto transform_node = m_assembly.surface_kinematics()[m];
            const auto& colloc_h_m = colloc_h.views()[m];
            const auto& velocities_h_m = velocities_h.views()[m];
            const auto& velocities_m = velocities.views()[m];
            const auto mat_transform_dual = transform_node->transform_dual(t_final);

            for (i64 j = 0; j < colloc_h_m.shape(1); j++) {
                for (i64 i = 0; i < colloc_h_m.shape(0); i++) {
                    auto local_velocity = -transform_node->linear_velocity(mat_transform_dual, {colloc_h_m(i, j, 0), colloc_h_m(i, j, 1), colloc_h_m(i, j, 2)});
                    velocities_h_m(i, j, 0) = local_velocity.x;
                    velocities_h_m(i, j, 1) = local_velocity.y;
                    velocities_h_m(i, j, 2) = local_velocity.z;
                }
            }
            velocities_h_m.to(velocities_m);
        }

        auto rhs_s = rhs.view().slice(All, s);
        backend->rhs_assemble_velocities(rhs_s, normals_d.views(), velocities.views());
    }
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

    std::printf("t_final: %f\n", t_final);

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
        // simulation.run(t_final, omega);
    }
    return 0;
}
#include <vector>
#include <string>
#include <functional> // std::function
#include <fstream>

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
        Tensor2D<Location::Device> q_mat{backend->memory.get()}; // (ns*nc) x harmonics
        Tensor2D<Location::Host> q_mat_h{backend->memory.get()}; // (ns*nc) x harmonics

        Tensor2D<Location::Device> residual{backend->memory.get()}; // (ns*nc) x harmonics
        Tensor2D<Location::Host> dft_h{backend->memory.get()}; // harmonics x harmonics
        Tensor2D<Location::Device> dft_d{backend->memory.get()};
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
    Simulation(backend_name, assembly.mesh_filenames(), true),
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
    const i64 unknowns = 2 * m_harmonics + 1;
    // Mesh
    MultiDim<3> gamma;
    MultiDim<3> panels_3D;  
    MultiDim<2> panels_2D;
    MultiDim<3> verts_wing_3D;
    MultiDim<2> transforms_2D;
    for (const auto& [ns, nc] : assembly_wings) {
        gamma.push_back({ns, nc, unknowns});
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
    rhs.init({n, unknowns});
    q_mat.init({n, unknowns});
    q_mat_h.init({n, unknowns});
    residual.init({n, unknowns});
    dft_d.init({unknowns, unknowns});
    dft_h.init({unknowns, unknowns});
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

    for (const auto& [wing_init, wing] : zip(verts_wing_init.views(), verts_wing.views())) wing_init.to(wing);
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);

    // Compute the fixed time step
    const auto& verts_first_wing = verts_wing_init_h.views()[0];
    const f32 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    const f32 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 1);
    const f32 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 2);
    const f32 last_panel_chord = std::sqrt(dx*dx + dy*dy + dz*dz);
    const f32 dt = last_panel_chord / linalg::length(m_assembly.kinematics()->linear_velocity(0.0f, {0.f, 0.f, 0.f}));
    const i64 max_t_steps = static_cast<i64>((t_start+period+dt) / dt);

    // Allocate the wake geometry
    {
        MultiDim<2> wake_panels_2D;
        MultiDim<3> verts_wake_3D;
        for (const auto& [ns, nc] : assembly_wings) {
            wake_panels_2D.push_back({ns, max_t_steps-1});
            verts_wake_3D.push_back({ns+1, max_t_steps, 4});
        }
        verts_wake.init(verts_wake_3D);
    }

    // Precompute constant lhs since we have a rigid wing
    lhs.view().fill(0.f);
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0 , 0);
    solver->factorize(lhs.view());

    // Pregenerate wake
    for (i32 i = 0; i < max_t_steps; i++) {
        const f32 t = (f32)i * dt;

        // parallel for
        for (i64 m = 0; m < assembly_wings.size(); m++) {
            auto transform = m_assembly.surface_kinematics()[m]->transform(t);
            transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
            transforms_h.views()[m].to(transforms.views()[m]);
        }

        backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
        backend->wake_shed(verts_wing.views(), verts_wake.views(), i);

        if (i == 0) { // todo: check if this is necessary for the velocity computation
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
            for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);
        }
    }

    // Assemble DFT matrix
    {
        const i32 unknowns = 2 * m_harmonics + 1;
        const f32 sqrt_unknowns = 1.f / std::sqrt(static_cast<f32>(unknowns));
        const f32 sqrt_unknowns_2 = std::sqrt(2.f / static_cast<f32>(unknowns));
        auto& dft_hv = dft_h.view();
        auto& dft_dv = dft_d.view();
        for (i64 i = 0; i < unknowns; i++) {
            dft_hv(i, 0) = sqrt_unknowns;
        }
        for (i64 j = 1; j < unknowns; j += 2) {
            const f32 k = ((f32)j + 1.f) / 2.f;
            for (i64 i = 0; i < unknowns; i++) {
                const f32 tn = ((f32)i / (f32)unknowns) * period;
                dft_hv(i, j) = std::cos(omega * tn * k) * sqrt_unknowns_2;
                dft_hv(i, j+1) = std::sin(omega * tn * k) * sqrt_unknowns_2;
            }
        }
        dft_hv.to(dft_dv);
    }

    // Iterative process for solving the blocked harmonic balance equation
    auto& residual_v = residual.view();
    residual_v.fill(1.f);
    const f32 tol = 1e-5f;
    const i32 max_iter = 50;
    i32 iter = 0;
    while (backend->blas->norm(residual_v.reshape(residual_v.shape(0)*residual_v.shape(1))) > tol) {
        
        rhs.view().fill(0.f);
        // For each unknown we fill their respective rhs column in a matrix free fashion
        for (i32 s = 0; s < 2*m_harmonics+1; s++) {
            const f32 t_final = t_start + period * (f32)s / (f32)(2*m_harmonics+1);
            const i64 t_steps = static_cast<i64>(std::round(t_final / dt));
            const f32 t = (f32)t_steps * dt; // t_final rounded to nearest dt

            for (i64 m = 0; m < assembly_wings.size(); m++) {
                auto transform = m_assembly.surface_kinematics()[m]->transform(t);
                transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
                transforms_h.views()[m].to(transforms.views()[m]);
            }

            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());

            // parallel for
            for (i64 m = 0; m < assembly_wings.size(); m++) {
                const auto transform_node = m_assembly.surface_kinematics()[m];
                const auto& colloc_h_m = colloc_h.views()[m];
                const auto& velocities_h_m = velocities_h.views()[m];
                const auto& velocities_m = velocities.views()[m];
                const auto mat_transform_dual = transform_node->transform_dual(t);

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
            // TODO: compute wake gammas from gamma_wing fourier coefficients
            // backend->rhs_assemble_wake_influence(rhs_s, gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), m_assembly.lifting(), t_steps);
        } // unknowns for loop

        // Apply the the dft matrix to the rhs
        // A @ G @ D^T = R <=> A @ G = R @ D (since D^-1 = D^T)
        backend->blas->gemm(1.0f, rhs.view(), dft_d.view(), 0.0f, q_mat.view());
        // Solve to obtain the fourier coefficients for each panel
        solver->solve(lhs.view(), q_mat.view());
        break; // TEMPORARY (only for no wake influence debugging)

        iter++;
        if (iter > max_iter) {
            std::printf("Newton-Raphson process did not converge\n");
            break;
        }
    } // while loop

    // Compute time domain solution from the obtained fourier coeffs
    {
        auto& q_mat_hv = q_mat_h.view();
        q_mat.view().to(q_mat_hv);
        const i32 unknowns = 2 * m_harmonics + 1;
        const f32 sqrt_unknowns = 1.f / std::sqrt(static_cast<f32>(unknowns));
        const f32 sqrt_unknowns_2 = std::sqrt(2.f / static_cast<f32>(unknowns));
        std::ofstream gamma_data("hbvlm_gamma_" + backend->name + ".txt");
        for (i64 i = 0; i < max_t_steps; i++) {
            const f32 t = (f32)i * dt;
            f32 gamma = sqrt_unknowns * q_mat_hv(0,0);
            for (i64 j = 1; j < unknowns; j += 2) {
                const f32 k = ((f32)j + 1.f) / 2.f;
                gamma += q_mat_hv(0, j) * std::cos(omega * t * k) * sqrt_unknowns_2;
                gamma += q_mat_hv(0, j+1) * std::sin(omega * t * k) * sqrt_unknowns_2;
            }
            gamma_data << t << " " << gamma << "\n";
        }
    }
}

int main() {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_10x5.x"};
    const std::vector<std::string> backends = {"cpu"};

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 4.0f;
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
        HBVLM simulation{backend_name, assembly, harmonics};
        simulation.run(t_final, omega);
    }
    return 0;
}
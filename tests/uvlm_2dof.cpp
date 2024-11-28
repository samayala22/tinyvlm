#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"
#include "vlm_integrator.hpp"

#include "tinycombination.hpp"
#include "tinypbar.hpp"

#include <tuple>
#include <fstream>

using namespace vlm;
using namespace linalg::ostream_overloads;

struct UVLM_2DOF_Vars {
    // Independent Dimensional parameters
    f32 b; // half chord
    f32 k_a; // pitch spring stiffness (arbitrary value)
    f32 rho; // fluid density

    // IndependentNon-dimensional parameters
    f32 a_h; // distance from mid-chord to elastic axis (also called the pitch axis)
    f32 omega; // natural frequency ratio (omega = omega_h / omega_a)
    f32 zeta_a; // damping ratio of the pitch spring
    f32 zeta_h; // damping ratio of the heave spring
    f32 x_a; // distance from elastic axis to center of mass
    f32 mu; // mass ratio
    f32 r_a; // radius of gyration around elastic axis
    f32 U_a; // reduced freestream velocity
};

inline linalg::float4x4 dual_to_float(const KinematicMatrixDual& m) {
    return {
        {m.x.x.val(), m.x.y.val(), m.x.z.val(), m.x.w.val()},
        {m.y.x.val(), m.y.y.val(), m.y.z.val(), m.y.w.val()},
        {m.z.x.val(), m.z.y.val(), m.z.z.val(), m.z.w.val()},
        {m.w.x.val(), m.w.y.val(), m.w.z.val(), m.w.w.val()}
    };
}

constexpr i32 DOF = 2;

class UVLM_2DOF final: public Simulation {
    public:
        UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~UVLM_2DOF() = default;
        void run(const Assembly& assembly, const UVLM_2DOF_Vars& vars, f32 t_final);

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Host> colloc_h{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        MultiTensor2D<Location::Device> areas_d{backend->memory.get()};

        MultiTensor3D<Location::Device> verts_wing_pos{backend->memory.get()}; // (nc+1)*(ns+1)*3

        // Data
        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        MultiTensor2D<Location::Device> gamma_wing{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wake{backend->memory.get()}; // nw*ns
        MultiTensor2D<Location::Device> gamma_wing_prev{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wing_delta{backend->memory.get()}; // nc*ns
        MultiTensor3D<Location::Device> aero_forces{backend->memory.get()}; // ns*nc*3

        MultiTensor3D<Location::Device> velocities{backend->memory.get()}; // ns*nc*3
        MultiTensor3D<Location::Host> velocities_h{backend->memory.get()}; // ns*nc*3

        MultiTensor2D<Location::Host> transforms_h{backend->memory.get()};
        MultiTensor2D<Location::Device> transforms{backend->memory.get()};
        
        Tensor1D<Location::Host> t_h{backend->memory.get()};
        std::unique_ptr<LU> solver;
        std::unique_ptr<LU> accel_solver;

        std::vector<i32> condition0;
    
        // Structure
        Tensor2D<Location::Host> M_h{backend->memory.get()};
        Tensor2D<Location::Host> C_h{backend->memory.get()};
        Tensor2D<Location::Host> K_h{backend->memory.get()};
        Tensor2D<Location::Host> u_h{backend->memory.get()};
        Tensor2D<Location::Host> v_h{backend->memory.get()};
        Tensor2D<Location::Host> a_h{backend->memory.get()};
        Tensor2D<Location::Host> F_h{backend->memory.get()};
        Tensor1D<Location::Host> du_h{backend->memory.get()};

        Tensor2D<Location::Device> M_d{backend->memory.get()};
        Tensor2D<Location::Device> M_factorized_d{backend->memory.get()};
        Tensor2D<Location::Device> C_d{backend->memory.get()};
        Tensor2D<Location::Device> K_d{backend->memory.get()};
        Tensor2D<Location::Device> u_d{backend->memory.get()}; // dof x tsteps
        Tensor2D<Location::Device> v_d{backend->memory.get()}; // dof x tsteps
        Tensor2D<Location::Device> a_d{backend->memory.get()}; // dof x tsteps
        Tensor1D<Location::Device> du{backend->memory.get()}; // dof
        Tensor1D<Location::Device> du_k{backend->memory.get()}; // dof
        Tensor1D<Location::Device> dv{backend->memory.get()}; // dof
        Tensor1D<Location::Device> da{backend->memory.get()}; // dof
        Tensor1D<Location::Device> dF{backend->memory.get()}; // dof
        Tensor1D<Location::Host> dF_h{backend->memory.get()}; // dof

        NewmarkBeta integrator{backend.get()};
    private:
        void alloc_buffers();
        void update_wake_and_gamma(i64 iteration);
        void compute_local_velocities(const Assembly& assembly, f32 t);
        void update_transforms(const Assembly& assembly, f32 t);
};

UVLM_2DOF::UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    assert(meshes.size() == 1); // single body only
    solver = backend->create_lu_solver();
    accel_solver = backend->create_lu_solver();
    alloc_buffers();
}

inline i64 total_panels(const MultiDim<2>& assembly_wing) {
    i64 total = 0;
    for (const auto& wing : assembly_wing) {
        total += wing[0] * wing[1];
    }
    return total;
}

void UVLM_2DOF::alloc_buffers() {
    const i64 n = total_panels(assembly_wings);
    // Mesh
    MultiDim<3> panels_3D;
    MultiDim<2> panels_2D;
    MultiDim<3> verts_wing_3D;
    MultiDim<2> transforms_2D;
    for (const auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
        panels_2D.push_back({ns, nc});
        verts_wing_3D.push_back({ns+1, nc+1, 4});
        transforms_2D.push_back({4, 4});
    }
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    colloc_h.init(panels_3D);
    areas_d.init(panels_2D);
    verts_wing_pos.init(verts_wing_3D);

    // Data
    lhs.init({n, n});
    rhs.init({n});
    gamma_wing.init(panels_2D);
    gamma_wing_prev.init(panels_2D);
    gamma_wing_delta.init(panels_2D);
    velocities.init(panels_3D);
    velocities_h.init(panels_3D);
    aero_forces.init(panels_3D);
    transforms_h.init(transforms_2D);
    transforms.init(transforms_2D);
    solver->init(lhs.view());

    condition0.resize(assembly_wings.size()*assembly_wings.size());

    // Host tensors
    M_h.init({DOF,DOF});
    C_h.init({DOF,DOF});
    K_h.init({DOF,DOF});

    // Device tensors
    M_d.init({DOF,DOF});
    M_factorized_d.init({DOF,DOF});
    C_d.init({DOF,DOF});
    K_d.init({DOF,DOF});
    du.init({DOF});
    du_h.init({DOF});
    dv.init({DOF});
    da.init({DOF});
    du_k.init({DOF});
    dF.init({DOF});
    dF_h.init({DOF});

    accel_solver->init(M_d.view());
}

f32 torsional_func(f32 alpha) {
    return alpha;
}

void UVLM_2DOF::update_wake_and_gamma(i64 iteration) {
    i64 begin = 0;
    for (i64 m = 0; m < assembly_wings.size(); m++) {
        const auto& gamma_wing_i = gamma_wing.views()[m];
        const auto& gamma_wing_prev_i = gamma_wing_prev.views()[m];
        const auto& gamma_wing_delta_i = gamma_wing_delta.views()[m];
        const auto& gamma_wake_i = gamma_wake.views()[m];
        i64 end = begin + gamma_wing_i.size();

        // Store prev gamma before updating
        gamma_wing_i.to(gamma_wing_prev_i); 
        
        // Copy solution to gamma_wing
        rhs.view().slice(Range{begin, end}).to(gamma_wing_i.reshape(gamma_wing_i.size()));
        
        // Update gamma_delta
        gamma_wing_i.to(gamma_wing_delta_i);
        
        // Shed wake
        gamma_wing_i.slice(All, -1).to(gamma_wake_i.slice(All, -1-iteration));
        
        // Compute delta
        backend->blas->axpy(
            -1.0f,
            gamma_wing_i.slice(All, Range{0, -2}),
            gamma_wing_delta_i.slice(All, Range{1, -1})
        );
        
        begin = end;
    }
}

void UVLM_2DOF::compute_local_velocities(const Assembly& assembly, f32 t) {
    // parallel for
    for (i64 m = 0; m < assembly_wings.size(); m++) {
        const auto transform_node = assembly.surface_kinematics()[m];
        const auto& colloc_h_m = colloc_h.views()[m];
        const auto& velocities_h_m = velocities_h.views()[m];
        const auto& velocities_m = velocities.views()[m];
        const auto mat_transform_dual = transform_node->transform_dual(t);

        for (i64 j = 0; j < colloc_h_m.shape(1); j++) {
            for (i64 i = 0; i < colloc_h_m.shape(0); i++) {
                auto local_velocity = -transform_node->linear_velocity(mat_transform_dual, 
                    {colloc_h_m(i, j, 0), colloc_h_m(i, j, 1), colloc_h_m(i, j, 2)});
                velocities_h_m(i, j, 0) = local_velocity.x;
                velocities_h_m(i, j, 1) = local_velocity.y;
                velocities_h_m(i, j, 2) = local_velocity.z;
            }
        }
        velocities_h_m.to(velocities_m);
    }
}

inline linalg::float3 linear_velocity(const KinematicMatrixDual& transform_dual, const linalg::float3 vertex) {
    linalg::vec<fwd::Float, 4> new_pt = linalg::mul(transform_dual, {vertex.x, vertex.y, vertex.z, 1.0f});
    return {new_pt.x.grad(), new_pt.y.grad(), new_pt.z.grad()};
}

inline void wing_velocities(
    const KinematicMatrixDual& transform_dual,
    const TensorView3D<Location::Host>& colloc_h,
    const TensorView3D<Location::Host>& velocities_h,
    const TensorView3D<Location::Device>& velocities_d
) {
    for (i64 j = 0; j < colloc_h.shape(1); j++) {
        for (i64 i = 0; i < colloc_h.shape(0); i++) {
            auto local_velocity = -linear_velocity(transform_dual, 
                {colloc_h(i, j, 0), colloc_h(i, j, 1), colloc_h(i, j, 2)});
            // std::cout << "panel " << i << ": " << local_velocity << "\n";
            velocities_h(i, j, 0) = local_velocity.x;
            velocities_h(i, j, 1) = local_velocity.y;
            velocities_h(i, j, 2) = local_velocity.z;
        }
    }
    velocities_h.to(velocities_d);
}

void UVLM_2DOF::update_transforms(const Assembly& assembly, f32 t) {
    for (i64 m = 0; m < assembly_wings.size(); m++) {
        auto transform = assembly.surface_kinematics()[m]->transform(t);
        transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
        transforms_h.views()[m].to(transforms.views()[m]);
    }
}

void UVLM_2DOF::run(const Assembly& assembly, const UVLM_2DOF_Vars& vars, f32 t_final_nd) {
    const f32 m = vars.mu * (PI_f * vars.rho * vars.b*vars.b);
    const f32 omega_a = std::sqrt(vars.k_a / (m * (vars.r_a * vars.b)*(vars.r_a * vars.b)));
    const f32 U = vars.U_a * (vars.b * omega_a);
    const f32 t_final = t_final_nd * vars.b / U;

    std::printf("m: %f\n", m);
    std::printf("omega_a: %f\n", omega_a);
    std::printf("U: %f\n", U);

    // Copy raw meshes to device
    for (const auto& [init_h, init_d] : zip(verts_wing_init_h.views(), verts_wing_init.views())) {
        init_h.to(init_d);
    }
    for (const auto& [kinematics, transform_h, transform_d] : zip(assembly.surface_kinematics(), transforms_h.views(), transforms.views())) {
        auto transform = kinematics->transform(0.0f);
        transform.store(transform_h.ptr(), transform_h.stride(1));
        transform_h.to(transform_d);
    }
    backend->displace_wing(transforms.views(), verts_wing_pos.views(), verts_wing_init.views());
    for (const auto& [wing, wing_pos] : zip(verts_wing.views(), verts_wing_pos.views())) wing_pos.to(wing);
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);

    // 1.  Compute the fixed time step
    const auto& verts_first_wing = verts_wing_init_h.views()[0];
    const f32 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    const f32 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 1);
    const f32 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 2);
    const f32 last_panel_chord = std::sqrt(dx*dx + dy*dy + dz*dz);
    const f32 dt = last_panel_chord / linalg::length(assembly.kinematics()->linear_velocity(0.0f, {0.f, 0.f, 0.f}));
    const f32 dt_nd = dt * U / vars.b;
    const i64 t_steps = static_cast<i64>(t_final / dt);

    std::cout << "dt: " << dt << "\n";
    std::cout << "dt_nd: " << dt_nd << "\n";
    std::cout << "t_steps: " << t_steps << "\n";

    // Initialize structural data
    const auto& M_hv = M_h.view();
    const auto& C_hv = C_h.view();
    const auto& K_hv = K_h.view();
    const auto& M_dv = M_d.view();
    const auto& C_dv = C_d.view();
    const auto& K_dv = K_d.view();

    M_hv(0, 0) = 1.0f;
    M_hv(1, 0) = vars.x_a / pow(vars.r_a, 2);
    M_hv(0, 1) = vars.x_a;
    M_hv(1, 1) = 1.0f;

    C_hv(0, 0) = 2.0f * vars.zeta_h * (vars.omega / vars.U_a);
    C_hv(1, 0) = 0.0f;
    C_hv(0, 1) = 0.0f;
    C_hv(1, 1) = 2.0f * vars.zeta_a / vars.U_a;

    // K_hv(0, 0) = pow(vars.omega / vars.U_a, 2);
    // K_hv(1, 0) = 0.0f;
    // K_hv(0, 1) = 0.0f;
    // K_hv(1, 1) = 1.0f / pow(vars.U_a, 2);
    K_hv.fill(0.0f); // linear stiffness moved to the rhs

    M_hv.to(M_dv);
    M_hv.to(M_factorized_d.view());
    C_hv.to(C_dv);
    K_hv.to(K_dv);

    integrator.init(M_dv, C_dv, K_dv, dt_nd);

    u_h.init({DOF, t_steps});
    v_h.init({DOF, t_steps});
    a_h.init({DOF, t_steps});
    u_d.init({DOF, t_steps});
    v_d.init({DOF, t_steps});
    a_d.init({DOF, t_steps});
    F_h.init({DOF, t_steps});

    // Necessary ?
    const auto& u_hv = u_h.view();
    const auto& v_hv = v_h.view();
    const auto& a_hv = a_h.view();
    const auto& u_dv = u_d.view();
    const auto& v_dv = v_d.view();
    const auto& a_dv = a_d.view();
    const auto& F_hv = F_h.view();
    u_hv.fill(0.0f);
    v_hv.fill(0.0f);
    a_hv.fill(0.0f);
    u_dv.fill(0.0f);
    v_dv.fill(0.0f);
    a_dv.fill(0.0f);
    F_hv.fill(0.0f);

    // Initial condition
    u_hv(1, 0) = to_radians(3.f);
    u_hv.slice(All, 0).to(u_dv.slice(All, 0));
    v_hv.slice(All, 0).to(v_dv.slice(All, 0));
    
    // Compute initial acceleration
    a_hv(0, 0) = - pow(vars.omega / vars.U_a, 2) * u_hv(0, 0);
    a_hv(1, 0) = - 1.0f / (vars.U_a * vars.U_a) * torsional_func(u_hv(1, 0));
    a_hv.slice(All, 0).to(a_d.view().slice(All, 0));
    backend->blas->gemv(-1.0f, C_d.view(), v_dv.slice(All, 0), 1.0f, a_dv.slice(All, 0));
    backend->blas->gemv(-1.0f, K_d.view(), u_dv.slice(All, 0), 1.0f, a_dv.slice(All, 0));
    accel_solver->factorize(M_factorized_d.view());
    accel_solver->solve(M_factorized_d.view(), a_dv.slice(All, 0));
    a_dv.slice(All, 0).to(a_hv.slice(All, 0));

    // std::cout << "M: " << M_hv << "\n";
    // std::cout << "C: " << C_hv << "\n";
    // std::cout << "K: " << K_hv << "\n";
    // std::cout << u_hv.slice(All, 0) << "\n";
    // std::cout << v_hv.slice(All, 0) << "\n";
    // std::cout << a_hv.slice(All, 0) << "\n";
    
    // 2. Allocate the wake geometry
    {
        MultiDim<2> wake_panels_2D;
        MultiDim<3> verts_wake_3D;
        for (const auto& [ns, nc] : assembly_wings) {
            wake_panels_2D.push_back({ns, t_steps-1});
            verts_wake_3D.push_back({ns+1, t_steps, 4});
        }
        gamma_wake.init(wake_panels_2D);
        verts_wake.init(verts_wake_3D);
        t_h.init({t_steps});
    }

    // Precomputation (only valid in single body case)
    lhs.view().fill(0.f);
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0 , 0);
    solver->factorize(lhs.view());
    backend->wake_shed(verts_wing.views(), verts_wake.views(), 0);
    compute_local_velocities(assembly, 0.0f);
    rhs.view().fill(0.f);
    backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
    backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), 0);
    solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
    update_wake_and_gamma(0);

    std::ofstream data_2dof("2dof.txt");
    data_2dof << std::to_string(vars.U_a) << "\n"; // TODO: this should be according to inputs
    data_2dof << std::to_string(0.0f) << " " << u_hv(0,0) << " " << u_hv(1,0) << "\n";

    // Transient simulation loop
    for (i32 i = 0; i < t_steps-2; i++) {
        const f32 t = (f32)i * dt;
        const f32 t_nd = (f32)i * dt_nd;
        t_h.view()(i) = t;

        dF.view().fill(0.f);
        du.view().fill(0.f);
        du_k.view().fill(1.f);
        i32 iteration = 0;
        const i32 max_iter = 50;

        while (backend->blas->norm(du_k.view()) > 1e-7f) {
            du.view().to(du_k.view());
            integrator.step(
                M_dv,
                C_dv,
                v_dv.slice(All, i),
                a_dv.slice(All, i),
                du.view(),
                dv.view(),
                da.view(),
                dF.view(),
                dt_nd // dimensionless dt
            );
            du.view().to(du_h.view());
            u_dv.slice(All, i).to(u_dv.slice(All, i+1));
            v_dv.slice(All, i).to(v_dv.slice(All, i+1));
            a_dv.slice(All, i).to(a_dv.slice(All, i+1));
            backend->blas->axpy(1.0f, du.view(), u_dv.slice(All, i+1));
            backend->blas->axpy(1.0f, dv.view(), v_dv.slice(All, i+1));
            backend->blas->axpy(1.0f, da.view(), a_dv.slice(All, i+1));
            u_dv.slice(All, i+1).to(u_hv.slice(All, i+1));
            v_dv.slice(All, i+1).to(v_hv.slice(All, i+1));
            a_dv.slice(All, i+1).to(a_hv.slice(All, i+1));

            // std::cout << "h: " << u_hv(0, i+1) << " dh: " << v_hv(0, i+1) << " ddh: " << a_hv(0, i+1) << "\n";
            // std::cout << "a: " << u_hv(1, i+1) << " da: " << v_hv(1, i+1) << " dda: " << a_hv(1, i+1) << "\n";

            // Returns dimensionalized values
            const KinematicNode heave([&](const fwd::Float& t) {
                return translation_matrix<fwd::Float>({
                    0.0f,
                    0.0f,
                    u_hv(0, i+1) * vars.b + (v_hv(0, i+1) * U) * t
                });
            });
            const KinematicNode pitch([&](const fwd::Float& t) {
                return rotation_matrix<fwd::Float>(
                    {
                        vars.b + vars.a_h * vars.b,
                        0.0f,
                        0.0f
                    },
                    {
                        0.0f,
                        1.0f,
                        0.0f
                    }, 
                    (u_hv(1, i+1) + (v_hv(1, i+1) * U / vars.b) * t)
                );
            });
            
            // Manually compute the transform (update_transforms(assembly, t+dt))
            auto fs = assembly.surface_kinematics()[0]->transform_dual(t+dt);
            auto transform_dual = linalg::mul(linalg::mul(fs, heave.transform_dual(0.0f)), pitch.transform_dual(0.0f));
            auto transform = dual_to_float(transform_dual);
            transform.store(transforms_h.views()[0].ptr(), transforms_h.views()[0].stride(1));
            transforms_h.views()[0].to(transforms.views()[0]);

            // Aero
            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_pos.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
            backend->wake_shed(verts_wing.views(), verts_wake.views(), i+1);
            
            wing_velocities(transform_dual, colloc_h.views()[0], velocities_h.views()[0], velocities.views()[0]);
            
            rhs.view().fill(0.f);
            backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
            backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), i+1);
            solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
            update_wake_and_gamma(i+1);

            const linalg::float3 freestream = -assembly.kinematics()->linear_velocity(t+dt, {0.f, 0.f, 0.f});
            backend->forces_unsteady(
                verts_wing.views()[0],
                gamma_wing_delta.views()[0],
                gamma_wing.views()[0],
                gamma_wing_prev.views()[0],
                velocities.views()[0],
                areas_d.views()[0],
                normals_d.views()[0],
                aero_forces.views()[0],
                dt
            );

            const f32 cl = backend->coeff_cl(
                aero_forces.views()[0],
                linalg::normalize(linalg::cross(freestream, {0.f, 1.f, 0.f})), // lift axis
                freestream,
                vars.rho,
                backend->sum(areas_d.views()[0])
            );

            auto ref_pt_4 = linalg::mul(transform, {vars.b + vars.a_h * vars.b, 0.0f, 0.0f, 1.0f});

            const linalg::float3 cm = backend->coeff_cm(
                aero_forces.views()[0],
                verts_wing.views()[0],
                {ref_pt_4.x, ref_pt_4.y, ref_pt_4.z},
                freestream,
                vars.rho,
                backend->sum(areas_d.views()[0]),
                backend->mesh_mac(verts_wing.views()[0], areas_d.views()[0])
            );

            std::printf("iter: %d | cl: %f | cm: %f\n", iteration, cl, cm.y);

            F_hv(0, i+1) = - cl / (PI_f * vars.mu);
            F_hv(1, i+1) = (2*cm.y) / (PI_f * vars.mu * pow(vars.r_a, 2));
 
            dF_h.view()(0) = F_hv(0, i+1) - F_hv(0, i) - pow(vars.omega / vars.U_a, 2) * du_h.view()(0);
            dF_h.view()(1) = F_hv(1, i+1) - F_hv(1, i) - 1/pow(vars.U_a, 2) * (torsional_func(u_hv(1,i+1)) - torsional_func(u_hv(1,i)));
            dF_h.view().to(dF.view());

            backend->blas->axpy(-1.0f, du.view(), du_k.view());
            iteration++;
            if (iteration > max_iter) {
                std::printf("Newton process did not converge\n");
                break;
            }
        }

        data_2dof << t_nd+dt_nd << " " << u_hv(0,i+1) << " " << u_hv(1,i+1) << "\n";
        
        std::printf("%d| iters: %d\n", i, iteration);

        // Classic UVLM:
        // update_transforms(assembly, t+dt);
        // backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_pos.views());
        // backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
        // backend->wake_shed(verts_wing.views(), verts_wake.views(), i+1);
        // compute_local_velocities(assembly, t+dt);
        // rhs.view().fill(0.f);
        // backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
        // backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), i+1);
        // solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
        // update_wake_and_gamma(i+1);

        // const linalg::float3 freestream = -assembly.kinematics()->linear_velocity(t+dt, {0.f, 0.f, 0.f});
        // backend->forces_unsteady(
        //     verts_wing.views()[0],
        //     gamma_wing_delta.views()[0],
        //     gamma_wing.views()[0],
        //     gamma_wing_prev.views()[0],
        //     velocities.views()[0],
        //     areas_d.views()[0],
        //     normals_d.views()[0],
        //     aero_forces.views()[0],
        //     dt
        // );

        // const f32 cl = backend->coeff_cl(
        //     aero_forces.views()[0],
        //     linalg::normalize(linalg::cross(freestream, {0.f, 1.f, 0.f})), // lift axis
        //     freestream,
        //     vars.rho,
        //     backend->sum(areas_d.views()[0])
        // );
    }
}

int main() {
    // vlm::Executor::instance(1);
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_2x2.x"};
    // const std::vector<std::string> meshes = {"../../../../mesh/rectangular_4x4.x"};
    const std::vector<std::string> backends = {"cpu"};

    auto simulations = tiny::make_combination(meshes, backends);

    const f32 flutter_speed = 6.285f;
    const f32 flutter_ratio = 0.2f;
    const f32 t_final_nd = 60.f;

    UVLM_2DOF_Vars vars;
    vars.b = 0.5f;
    vars.k_a = 1000.0f;
    vars.rho = 1.0f;
    vars.a_h = -0.5f;
    vars.omega = 0.2f;
    vars.zeta_a = 0.0f;
    vars.zeta_h = 0.0f;
    vars.x_a = 0.25f;
    vars.mu = 100.0f;
    vars.r_a = 0.5f;
    vars.U_a = flutter_ratio * flutter_speed;

    const f32 m = vars.mu * (PI_f * vars.rho * vars.b*vars.b);
    const f32 omega_a = std::sqrt(vars.k_a / (m * (vars.r_a * vars.b)*(vars.r_a * vars.b)));
    const f32 U = vars.U_a * (vars.b * omega_a);

    KinematicsTree kinematics_tree;
    
    auto freestream = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            -U*t,
            0.0f,
            0.0f
        });
    });

    for (const auto& [mesh_name, backend_name] : simulations) {
        Assembly assembly(freestream);
        assembly.add(mesh_name, freestream);
        UVLM_2DOF simulation{backend_name, {mesh_name}};
        simulation.run(assembly, vars, t_final_nd);
    }
    return 0;
}
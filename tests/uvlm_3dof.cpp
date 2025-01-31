#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"
#include "vlm_integrator.hpp"
#include "vlm_io.hpp"

#include "tinycombination.hpp"
#include "tinypbar.hpp"

#include <fstream>
#include <memory>

#define ASSERT_EQ(x, y) \
    do { \
        auto val1 = (x); \
        auto val2 = (y); \
        if (!(val1 == val2)) { \
            std::cerr << "Assertion failed: " << #x << " == " << #y << " (Left: " << val1 << ", Right: " << val2 << ")\n"; \
            std::abort(); \
        } \
    } while (0)

#define ASSERT_NEAR(x, y, tol) \
    do { \
        auto val1 = (x); \
        auto val2 = (y); \
        if (!(std::abs(val1 - val2) <= tol)) { \
            std::cerr << "Assertion failed: |" << #x << " - " << #y << "| <= " << tol << " (Left: " << val1 << ", Right: " << val2 << ", Diff: " << std::abs(val1 - val2) << ")\n"; \
            std::abort(); \
        } \
    } while (0)

using namespace vlm;
using namespace linalg::ostream_overloads;

// Notes:
// 1. Nonlinear term is defined twice (structural initialization and in the coupling loop)
// 2. Initial position is defined twice (strucrual initialization and verts_init position)

struct Vars {
    f32 a; // Dimensionless distance between mid-chord and EA (-0.5)
    f32 b; // Semi-chord (0.127 m)
    f32 c; // Dimensionless distance between flap hinge and mid-chord (0.5); Plunge structural damping coefficient per unit span (1.7628 kg/ms)
    f32 I_alpha; // Mass moment of inertia of the wing-flap about wing EA per unit span (0.01347 kgm)
    f32 I_beta; // Mass moment of inertia of the flap about the flap hinge line per unit span (0.0003264 kgm)
    f32 k_h; // Linear structural stiffness coefficient of plunging per unit span
    f32 k_alpha; // Linear structural stiffness coefficient of plunging per unit span (2818.8 kg/ms²)
    f32 k_beta; // Linear structural stiffness coefficient of pitching per unit span (37.34 kgm/s²); Linear structural stiffness coefficient of flap per unit span (3.9 kgm/s²); Mass of wing-aileron per span (1.558 kg/m)
    f32 m; // Mass of wing-aileron per span
    f32 m_t; // Mass of wing-aileron and the supports per span
    f32 r_alpha; // Dimensionless radius of gyration around elastic axis
    f32 r_beta; // Dimensionless radius of gyration around flap hinge axis
    f32 S_alpha; // Static mass moment of wing-flap about wing EA per unit span
    f32 S_beta; // Static mass moment of flap about flap hinge line per unit span
    f32 x_alpha; // Dimensionless distance between airfoil EA and the center of gravity
    f32 x_beta; // Dimensionless distance between flap center of gravity and flap hinge axis
    f32 omega_h; // Uncoupled plunge natural frequency
    f32 omega_alpha; // Uncoupled pitch natural frequency
    f32 omega_beta; // Uncoupled flap natural frequency
    f32 rho; // Fluid density
    f32 zeta_h; // Plunge damping ratio
    f32 zeta_alpha; // Pitch damping ratio
    f32 zeta_beta; // Flap damping ratio
    f32 U; // Velocity
};

inline linalg::float4x4 dual_to_float(const KinematicMatrixDual& m) {
    return {
        {m.x.x.val(), m.x.y.val(), m.x.z.val(), m.x.w.val()},
        {m.y.x.val(), m.y.y.val(), m.y.z.val(), m.y.w.val()},
        {m.z.x.val(), m.z.y.val(), m.z.z.val(), m.z.w.val()},
        {m.w.x.val(), m.w.y.val(), m.w.z.val(), m.w.w.val()}
    };
}

constexpr i32 DOF = 3;

class UVLM_3DOF final: public Simulation {
    public:
        UVLM_3DOF(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~UVLM_3DOF() = default;
        void run(const Assembly& assembly, const Vars& vars, f32 t_final);

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Host> colloc_h{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        MultiTensor2D<Location::Device> areas_d{backend->memory.get()};
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
        void update_transforms(const Assembly& assembly, f32 t);
        void initialize_structure_data(const Vars& vars, const i64 t_steps, const f32 dt_nd);
        f32 wing_alpha(); // get pitch angle from verts_wing_h
        f32 flap_beta();
        f32 wing_h(f32 elastic_dst); // get wing elastic height at given dist
};

UVLM_3DOF::UVLM_3DOF(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes, false) {
    ASSERT_EQ(meshes.size(), 2);
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

f32 UVLM_3DOF::wing_alpha() { // angle relative to the x axis
    const auto& wing_h = verts_wing_h.views()[0];
    const linalg::float3 chord_axis = linalg::normalize(linalg::float3{
        wing_h(0, 1, 0) - wing_h(0, 0, 0),
        wing_h(0, 1, 1) - wing_h(0, 0, 1),
        wing_h(0, 1, 2) - wing_h(0, 0, 2)
    });
    return std::atan2(-chord_axis.z, chord_axis.x);
}

f32 UVLM_3DOF::flap_beta() { // angle relative to the wing chord axis
    const auto& flap_h = verts_wing_h.views()[1];
    const linalg::float3 chord_axis = linalg::normalize(linalg::float3{
        flap_h(0, 1, 0) - flap_h(0, 0, 0),
        flap_h(0, 1, 1) - flap_h(0, 0, 1),
        flap_h(0, 1, 2) - flap_h(0, 0, 2)
    });
    return std::atan2(-chord_axis.z, chord_axis.x) - wing_alpha();
}

// elastic dst: distance from the leading edge
f32 UVLM_3DOF::wing_h(f32 elastic_dst) {
    const auto& wing_h = verts_wing_h.views()[0];
    const linalg::float3 pt0 = linalg::float3{wing_h(0, 0, 0), wing_h(0, 0, 1), wing_h(0, 0, 2)};
    const linalg::float3 pt1 = linalg::float3{wing_h(0, 1, 0), wing_h(0, 1, 1), wing_h(0, 1, 2)};
    const linalg::float3 chord_axis = linalg::normalize(pt1 - pt0);
    const linalg::float3 elastic_pt = pt0 + elastic_dst * chord_axis;
    return elastic_pt.z;
}

void UVLM_3DOF::alloc_buffers() {
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

// f32 alpha(f32 alpha) {
//     return alpha;
// }

f32 alpha_freeplay(f32 alpha, f32 M0 = 0.0f, f32 Mf = 0.0f, f32 delta = to_radians(4.24f), f32 a_f = to_radians(-2.12f)) {
    if (alpha < a_f) {
        return M0 + alpha - a_f;
    } else if (alpha >= a_f && alpha <= (a_f + delta)) {
        return M0 + Mf * (alpha - a_f);
    } else { // alpha > a_F + delta
        return M0 + alpha - a_f + delta * (Mf - 1);
    }
}

void UVLM_3DOF::update_wake_and_gamma(i64 iteration) {
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
        if (m == 1) { // only shed for the flap
            gamma_wing_i.slice(All, -1).to(gamma_wake_i.slice(All, -1-iteration));
        }
        
        // Compute delta
        backend->blas->axpy(
            -1.0f,
            gamma_wing_i.slice(All, Range{0, -2}),
            gamma_wing_delta_i.slice(All, Range{1, -1})
        );
        
        begin = end;
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

void UVLM_3DOF::update_transforms(const Assembly& assembly, f32 t) {
    for (i64 m = 0; m < assembly_wings.size(); m++) {
        auto transform = assembly.surface_kinematics()[m]->transform(t);
        transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
        transforms_h.views()[m].to(transforms.views()[m]);
    }
}

void UVLM_3DOF::initialize_structure_data(const Vars& v, const i64 t_steps, const f32 dt_nd) {
    // Initialize structural data
    const auto& M_hv = M_h.view();
    const auto& C_hv = C_h.view();
    const auto& K_hv = K_h.view();
    const auto& M_dv = M_d.view();
    const auto& C_dv = C_d.view();
    const auto& K_dv = K_d.view();

    M_hv.fill(0.0f);
    C_hv.fill(0.0f);
    K_hv.fill(0.0f);

    const f32 sigma = v.omega_h / v.omega_alpha;

    M_hv(0, 0) = v.m_t / v.m;
    M_hv(0, 1) = v.x_alpha;
    M_hv(0, 2) = v.x_beta;
    M_hv(1, 0) = v.x_alpha;
    M_hv(1, 1) = pow(v.r_alpha, 2);
    M_hv(1, 2) = (v.c - v.a) * v.x_beta + pow(v.r_beta, 2);
    M_hv(2, 0) = v.x_beta;
    M_hv(2, 1) = (v.c - v.a) * v.x_beta + pow(v.r_beta, 2);
    M_hv(2, 2) = pow(v.r_beta, 2);

    C_hv(0, 0) = 2.f * (sigma * v.zeta_h);
    C_hv(1, 1) = 2.f * (pow(v.r_alpha, 2) * v.zeta_alpha);
    C_hv(2, 2) = 2.f * ((v.omega_beta / v.omega_alpha) * pow(v.r_beta, 2) * v.zeta_beta);

    K_hv(1, 1) = pow(v.r_alpha, 2);

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
    u_hv(0, 0) = 0.01f / v.b;
    // Transfer to device
    u_hv.slice(All, 0).to(u_dv.slice(All, 0));
    v_hv.slice(All, 0).to(v_dv.slice(All, 0));
    
    // Compute initial acceleration
    // const f32 gamma = 0.0f;
    // a_hv(0, 0) = - u_hv(0, 0) * pow(sigma, 2) * (1 + gamma * pow(u_hv(0,0), 2));
    a_hv(0, 0) = - u_hv(0, 0) * pow(sigma, 2);
    a_hv(2, 0) = - (pow(v.omega_beta / v.omega_alpha, 2) * pow(v.r_beta, 2)) * alpha_freeplay(u_hv(2, 0));
    
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
}

void nondimensionalize_verts(TensorView3D<Location::Host>& verts, const f32 b) {
    for (i64 j = 0; j < verts.shape(1); j++) {
        for (i64 i = 0; i < verts.shape(0); i++) {
            verts(i, j, 0) /= b;
            verts(i, j, 1) /= b;
            verts(i, j, 2) /= b;
        }
    }
}

void UVLM_3DOF::run(const Assembly& assembly, const Vars& v, f32 t_final_nd) {
    const f32 V = v.U / (v.b * v.omega_alpha);
    const f32 mu = v.m / (PI_f * v.rho * v.b*v.b);

    nondimensionalize_verts(verts_wing_init_h.views()[0], v.b);
    nondimensionalize_verts(verts_wing_init_h.views()[1], v.b);

    verts_wing_init_h.views()[0].to(verts_wing_init.views()[0]);
    verts_wing_init_h.views()[1].to(verts_wing_init.views()[1]);

    auto init_pos = translation_matrix<f32>(
        {
            0.0f,
            0.0f,
            - 0.01f / v.b
        }
    );

    for (const auto& [transform_h, transform_d] : zip(transforms_h.views(), transforms.views())) {
        init_pos.store(transform_h.ptr(), transform_h.stride(1));
        transform_h.to(transform_d);
    }
    backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);

    // 1.  Compute the fixed time step
    const auto& verts_first_wing = verts_wing_init_h.views()[1];
    const f32 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    const f32 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 1);
    const f32 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 2);
    const f32 last_panel_chord = std::sqrt(dx*dx + dy*dy + dz*dz);
    ASSERT_NEAR(linalg::length(assembly.kinematics()->linear_velocity(0.0f, {0.f, 0.f, 0.f})), v.U / (v.b * v.omega_alpha), 1e-6f);
    const f32 dt_nd = last_panel_chord / linalg::length(assembly.kinematics()->linear_velocity(0.0f, {0.f, 0.f, 0.f}));
    const i64 t_steps = static_cast<i64>(t_final_nd / dt_nd);
    t_h.init({t_steps-2});

    std::printf("dt: %f\n", dt_nd);
    std::printf("t_steps: %lld\n", t_steps);

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
        verts_wake_h.init(verts_wake_3D);
    }

    initialize_structure_data(v, t_steps, dt_nd);

    verts_wake_h.views()[0].fill(0.0f); // Unused
    verts_wake_h.views()[1].fill(0.0f);

    // Precomputation (only valid in single body case)
    lhs.view().fill(0.f);
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0 , 0);
    solver->factorize(lhs.view());
    backend->wake_shed(verts_wing.views(), verts_wake.views(), 0);
    // Technically in the initial velocities calculation we should also take into account the IC of the eq of motion
    // In our case, since the IC is an initial position, we can ignore it. 
    wing_velocities(assembly.surface_kinematics()[0]->transform_dual(0.0f), colloc_h.views()[0], velocities_h.views()[0], velocities.views()[0]);
    wing_velocities(assembly.surface_kinematics()[0]->transform_dual(0.0f), colloc_h.views()[1], velocities_h.views()[1], velocities.views()[1]);

    rhs.view().fill(0.f);
    backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
    backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), 0);
    solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
    update_wake_and_gamma(0);

    std::ofstream data_3dof("3dof.txt");
    data_3dof << 1.0f << "\n";

    // Transient simulation loop
    // for (i32 i = 0; i < t_steps-2; i++) {
    for (const i32 i : tiny::pbar(0, (i32)t_steps-2)) {

        const f32 t_nd = (f32)i * dt_nd;
        t_h.view()(i) = t_nd;

        data_3dof << t_nd 
            << " " << u_h.view()(0,i) // h
            << " " << u_h.view()(1,i) // alpha
            << " " << u_h.view()(2,i) // beta
            << " " << v_h.view()(0,i) // dh/dt
            << " " << v_h.view()(1,i) // dalpha/dt
            << " " << v_h.view()(2,i) // dbeta/dt
            << " " << F_h.view()(0,i)
            << " " << F_h.view()(1,i)
            << " " << F_h.view()(2,i)
            << "\n";

        for (const auto& [gamma_wing_i, gamma_wing_prev_i] : zip(gamma_wing.views(), gamma_wing_prev.views())) {
            gamma_wing_i.to(gamma_wing_prev_i);
        }
        
        dF.view().fill(0.f);
        du.view().fill(0.f);
        du_k.view().fill(1.f);
        i32 iteration = 0;
        const i32 max_iter = 50;

        while (backend->blas->norm(du_k.view()) > EPS_f) {
            du.view().to(du_k.view());
            integrator.step(
                M_d.view(),
                C_d.view(),
                v_d.view().slice(All, i),
                a_d.view().slice(All, i),
                du.view(),
                dv.view(),
                da.view(),
                dF.view(),
                dt_nd // dimensionless dt
            );

            verts_wing.views()[0].to(verts_wing_h.views()[0]);
            verts_wing.views()[1].to(verts_wing_h.views()[1]);
            
            // Temporary Position correctness checks
            const f32 real_alpha = wing_alpha();
            const f32 real_beta = flap_beta();
            const f32 real_h = wing_h(1 + v.a);
            ASSERT_NEAR(real_h, -u_h.view()(0, i), 1e-5f);
            ASSERT_NEAR(real_alpha, u_h.view()(1, i), 1e-5f);
            ASSERT_NEAR(real_beta, u_h.view()(2, i), 1e-5f);

            du.view().to(du_h.view());

            u_d.view().slice(All, i).to(u_d.view().slice(All, i+1));
            v_d.view().slice(All, i).to(v_d.view().slice(All, i+1));
            a_d.view().slice(All, i).to(a_d.view().slice(All, i+1));
            backend->blas->axpy(1.0f, du.view(), u_d.view().slice(All, i+1));
            backend->blas->axpy(1.0f, dv.view(), v_d.view().slice(All, i+1));
            backend->blas->axpy(1.0f, da.view(), a_d.view().slice(All, i+1));
            u_d.view().slice(All, i+1).to(u_h.view().slice(All, i+1));
            v_d.view().slice(All, i+1).to(v_h.view().slice(All, i+1));
            a_d.view().slice(All, i+1).to(a_h.view().slice(All, i+1));

            const KinematicNode heave([&](const fwd::Float& t) {
                return translation_matrix<fwd::Float>({
                    0.0f,
                    0.0f,
                    - u_h.view()(0, i+1) - v_h.view()(0, i+1) * t
                });
            });
            const KinematicNode wing_pitch([&](const fwd::Float& t) {
                return rotation_matrix<fwd::Float>(
                    {
                        1.f + v.a,
                        0.0f,
                        0.0f
                    },
                    {
                        0.0f,
                        1.0f,
                        0.0f
                    }, 
                    u_h.view()(1, i+1) + v_h.view()(1, i+1) * t
                );
            });
            const KinematicNode flap_pitch([&](const fwd::Float& t) {
                return rotation_matrix<fwd::Float>(
                    {
                        1.f + v.c,
                        0.0f,
                        0.0f
                    },
                    {
                        0.0f,
                        1.0f,
                        0.0f
                    }, 
                    u_h.view()(2, i+1) + v_h.view()(2, i+1) * t
                );
            }); 

            // std::printf("iter: %d | h: %f | alpha: %f | h_dot: %f | alpha_dot: %f\n", iteration, u_h.view()(0, i+1), u_h.view()(1, i+1), v_h.view()(0, i+1), v_h.view()(1, i+1));
            
            // Manually compute the transform (update_transforms(assembly, t+dt))
            auto fs = assembly.kinematics()->transform_dual(t_nd+dt_nd);
            auto wing_transform_dual = linalg::mul(fs, linalg::mul(heave.transform_dual(0.0f), wing_pitch.transform_dual(0.0f)));
            auto flap_transform_dual = linalg::mul(wing_transform_dual, flap_pitch.transform_dual(0.0f));
            auto wing_transform = dual_to_float(wing_transform_dual);
            auto flap_transform = dual_to_float(flap_transform_dual);
            auto ref_pt_a = linalg::mul(wing_transform, {1 + v.a, 0.0f, 0.0f, 1.0f});
            auto ref_pt_b = linalg::mul(flap_transform, {1 + v.c, 0.0f, 0.0f, 1.0f});

            wing_transform.store(transforms_h.views()[0].ptr(), transforms_h.views()[0].stride(1));
            flap_transform.store(transforms_h.views()[1].ptr(), transforms_h.views()[1].stride(1));
            transforms_h.views()[0].to(transforms.views()[0]);
            transforms_h.views()[1].to(transforms.views()[1]);

            // Aero
            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
            backend->wake_shed(verts_wing.views(), verts_wake.views(), i+1);

            wing_velocities(wing_transform_dual, colloc_h.views()[0], velocities_h.views()[0], velocities.views()[0]);
            wing_velocities(flap_transform_dual, colloc_h.views()[1], velocities_h.views()[1], velocities.views()[1]);

            rhs.view().fill(0.f);
            backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
            backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), i+1);
            solver->solve(lhs.view(), rhs.view());

            {
                i64 begin = 0;
                for (i64 m = 0; m < assembly_wings.size(); m++) {
                    const auto& gamma_wing_i = gamma_wing.views()[m];
                    const auto& gamma_wing_delta_i = gamma_wing_delta.views()[m];
                    const auto& gamma_wake_i = gamma_wake.views()[m];
                    i64 end = begin + gamma_wing_i.size();

                    // Copy solution to gamma_wing
                    rhs.view().slice(Range{begin, end}).to(gamma_wing_i.reshape(gamma_wing_i.size()));
                    
                    // Update gamma_delta
                    gamma_wing_i.to(gamma_wing_delta_i);
                    
                    // Shed wake
                    gamma_wing_i.slice(All, -1).to(gamma_wake_i.slice(All, -1-(i+1)));
                    
                    // Compute delta
                    backend->blas->axpy(
                        -1.0f,
                        gamma_wing_i.slice(All, Range{0, -2}),
                        gamma_wing_delta_i.slice(All, Range{1, -1})
                    );
                    
                    begin = end;
                }
            }

            const linalg::float3 freestream = -assembly.kinematics()->linear_velocity(t_nd+dt_nd, {0.f, 0.f, 0.f});
            backend->forces_unsteady_multibody(
                verts_wing.views(),
                gamma_wing_delta.views(),
                gamma_wing.views(),
                gamma_wing_prev.views(),
                velocities.views(),
                areas_d.views(),
                normals_d.views(),
                aero_forces.views(),
                dt_nd
            );

            const f32 cl = backend->coeff_cl_multibody(
                aero_forces.views(),
                areas_d.views(),
                freestream,
                1.0f // rho_inf = 1
            );

            const linalg::float3 cm_a = backend->coeff_cm_multibody(
                aero_forces.views(),
                verts_wing.views(),
                areas_d.views(),
                {ref_pt_a.x, ref_pt_a.y, ref_pt_a.z},
                freestream,
                1.0f // rho_inf = 1
            );
            const linalg::float3 cm_b = backend->coeff_cm_multibody(
                aero_forces.views(),
                verts_wing.views(),
                areas_d.views(),
                {ref_pt_b.x, ref_pt_b.y, ref_pt_b.z},
                freestream,
                1.0f // rho_inf = 1
            );
            
            // Aero forces
            F_h.view()(0, i+1) = - V*V*cl / (PI_f*mu);
            F_h.view()(1, i+1) = 2.f*V*V*cm_a.y / (PI_f*mu);
            F_h.view()(2, i+1) = 2.f*V*V*cm_b.y / (PI_f*mu);
            
            // deltaF for rhs
            const f32 sigma = v.omega_h / v.omega_alpha;
            dF_h.view()(0) = F_h.view()(0, i+1) - F_h.view()(0, i) - pow(sigma, 2) * du_h.view()(0);
            dF_h.view()(1) = F_h.view()(1, i+1) - F_h.view()(1, i);
            dF_h.view()(2) = F_h.view()(1, i+1) - F_h.view()(1, i) - (pow(v.omega_beta / v.omega_alpha, 2) * pow(v.r_beta, 2)) * (alpha_freeplay(u_h.view()(2,i+1)) - alpha_freeplay(u_h.view()(2,i)));

            dF_h.view().to(dF.view());

            backend->blas->axpy(-1.0f, du.view(), du_k.view());
            iteration++;
            if (iteration > max_iter) {
                std::printf("Newton process did not converge\n");
                break;
            }
        } // while loop
    } // simulation loop
}

int main() {
    // vlm::Executor::instance(1);
    const std::vector<std::vector<std::string>> meshes = {{
        "../../../../mesh/3dof_wing_9x5.x",
        "../../../../mesh/3dof_flap_3x5.x"
    }}; // vector of vector meh
    const std::vector<std::string> backends = {"cpu"};

    auto simulations = tiny::make_combination(meshes, backends);

    // const f32 flutter_speed = 23.9f;
    // const f32 flutter_ratio = 0.2f;

    Vars v;
    v.a = -0.5f;
    v.b = 0.127f;
    v.c = 0.5f;
    v.I_alpha = 0.01347f;
    v.I_beta = 0.0003264f;
    v.k_h = 2818.8f;
    v.k_alpha = 37.34f;
    v.k_beta = 3.9f;
    v.m = 1.5666f;
    v.m_t = 3.39298f;
    v.r_alpha = 0.7321f;
    v.r_beta = 0.1140f;
    v.S_alpha = 0.08587f;
    v.S_beta = 0.00395f;
    v.x_alpha = 0.4340f;
    v.x_beta = 0.02f;
    v.omega_h = 42.5352f;
    v.omega_alpha = 52.6506f;
    v.omega_beta = 109.3093f;
    v.rho = 1.225f;
    v.zeta_h = 0.0113f;
    v.zeta_alpha = 0.01626f;
    v.zeta_beta = 0.0115f;
    v.U = 12.36842f; // m/s

    const f32 t_final_nd = 10 * v.omega_alpha;

    KinematicsTree kinematics_tree;

    auto freestream = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            - (v.U / (v.b * v.omega_alpha)) * t,
            0.0f,
            0.0f
        });
    });

    for (const auto& [mesh_names, backend_name] : simulations) {
        Assembly assembly(freestream);
        assembly.add(mesh_names.get()[0], freestream);
        UVLM_3DOF simulation{backend_name, mesh_names};
        // simulation.run(assembly, v, t_final_nd);
    }
    
    return 0;
}
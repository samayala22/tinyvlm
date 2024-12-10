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

#define COUPLED 1

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
 
        MultiTensor<i8, 2, Location::Host> wing_cell_type{backend->memory.get()}; // ns*nc
        MultiTensor<i64, 2, Location::Host> wing_cell_offset{backend->memory.get()}; // ns*nc
        MultiTensor<i64, 3, Location::Host> wing_cell_connectivity{backend->memory.get()}; // 4x(ns*nc)

        MultiTensor<i8, 2, Location::Host> wake_cell_type{backend->memory.get()}; // ns x c
        MultiTensor<i64, 2, Location::Host> wake_cell_offset{backend->memory.get()}; // ns x nc
        MultiTensor<i64, 3, Location::Host> wake_cell_connectivity{backend->memory.get()}; // 4x(ns*nc)

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
        void initialize_structure_data(const UVLM_2DOF_Vars& vars, const i64 t_steps, const f32 dt_nd);
        f32 wing_alpha(); // get pitch angle from verts_wing_h
        f32 wing_h(f32 elastic_dst); // get wing elastic height at given dist
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

f32 UVLM_2DOF::wing_alpha() {
    const auto& wing_h = verts_wing_h.views()[0];
    const linalg::float3 chord_axis = linalg::normalize(linalg::float3{
        wing_h(0, 1, 0) - wing_h(0, 0, 0),
        wing_h(0, 1, 1) - wing_h(0, 0, 1),
        wing_h(0, 1, 2) - wing_h(0, 0, 2)
    });
    return std::atan2(-chord_axis.z, chord_axis.x);
}

// elastic dst: distance from the leading edge
f32 UVLM_2DOF::wing_h(f32 elastic_dst) {
    const auto& wing_h = verts_wing_h.views()[0];
    const linalg::float3 pt0 = linalg::float3{wing_h(0, 0, 0), wing_h(0, 0, 1), wing_h(0, 0, 2)};
    const linalg::float3 pt1 = linalg::float3{wing_h(0, 1, 0), wing_h(0, 1, 1), wing_h(0, 1, 2)};
    const linalg::float3 chord_axis = linalg::normalize(pt1 - pt0);
    const linalg::float3 elastic_pt = pt0 + elastic_dst * chord_axis - 0.25f * (pt1 - pt0);
    return elastic_pt.z;
}

void body_connectivity(
    const TensorView<f32, 3, Location::Host>& verts_wing_m,
    const TensorView<i8, 2, Location::Host>& cell_type_m,
    const TensorView<i64, 2, Location::Host>& cell_offset_m,
    const TensorView<i64, 3, Location::Host>& cell_connectivity_m
) {
    for (i64 j = 0; j < cell_type_m.shape(1); j++) {
        for (i64 i = 0; i < cell_type_m.shape(0); i++) {
            cell_type_m(i, j) = 9;
            cell_offset_m(i, j) = cell_offset_m.to_linear_index({i, j}) * 4 + 4;
            cell_connectivity_m(0, i, j) = verts_wing_m.to_linear_index({i+0, j+0, 0});
            cell_connectivity_m(1, i, j) = verts_wing_m.to_linear_index({i+1, j+0, 0});
            cell_connectivity_m(2, i, j) = verts_wing_m.to_linear_index({i+1, j+1, 0});
            cell_connectivity_m(3, i, j) = verts_wing_m.to_linear_index({i+0, j+1, 0});
        }
    }
}

void multibody_connectivity(
    const MultiTensorView<f32, 3, Location::Host>& verts_wing,
    const MultiTensorView<i8, 2, Location::Host>& cell_type,
    const MultiTensorView<i64, 2, Location::Host>& cell_offset,
    const MultiTensorView<i64, 3, Location::Host>& cell_connectivity
) {
    for (i64 m = 0; m < verts_wing.size(); m++) {
        const auto& verts_wing_m = verts_wing[m];
        const auto& cell_type_m = cell_type[m];
        const auto& cell_offset_m = cell_offset[m];
        const auto& cell_connectivity_m = cell_connectivity[m];
        body_connectivity(verts_wing_m, cell_type_m, cell_offset_m, cell_connectivity_m);
    }
}

void UVLM_2DOF::alloc_buffers() {
    const i64 n = total_panels(assembly_wings);
    // Mesh
    MultiDim<3> panels_3D;
    MultiDim<2> panels_2D;
    MultiDim<3> verts_wing_3D;
    MultiDim<2> transforms_2D;
    MultiDim<3> connectivity_3D;
    for (const auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
        panels_2D.push_back({ns, nc});
        verts_wing_3D.push_back({ns+1, nc+1, 4});
        transforms_2D.push_back({4, 4});
        connectivity_3D.push_back({4, ns, nc});
    }
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    colloc_h.init(panels_3D);
    areas_d.init(panels_2D);

    wing_cell_type.init(panels_2D);
    wing_cell_offset.init(panels_2D);
    wing_cell_connectivity.init(connectivity_3D);

    multibody_connectivity(verts_wing_h.views(), wing_cell_type.views(), wing_cell_offset.views(), wing_cell_connectivity.views());

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

void UVLM_2DOF::initialize_structure_data(const UVLM_2DOF_Vars& vars, const i64 t_steps, const f32 dt_nd) {
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

void UVLM_2DOF::run(const Assembly& assembly, const UVLM_2DOF_Vars& vars, f32 t_final_nd) {
    const f32 m = vars.mu * (PI_f * vars.rho * vars.b*vars.b);
    const f32 omega_a = std::sqrt(vars.k_a / (m * (vars.r_a * vars.b)*(vars.r_a * vars.b)));
    const f32 U = vars.U_a * (vars.b * omega_a);

    std::printf("m: %f\n", m);
    std::printf("omega_a: %f\n", omega_a);
    std::printf("U: %f\n", U);
    std::printf("U_a: %f\n", vars.U_a);

    nondimensionalize_verts(verts_wing_init_h.views()[0], vars.b);
    verts_wing_init_h.views()[0].to(verts_wing_init.views()[0]);

    #ifdef COUPLED
    auto init_pos = rotation_matrix<f32>(
        {
            1 + vars.a_h,
            0.0f,
            0.0f
        },
        {
            0.0f,
            1.0f,
            0.0f
        }, 
        to_radians(3.0f)
    );
    #else
    KinematicMatrix init_pos = linalg::identity;
    #endif

    for (const auto& [transform_h, transform_d] : zip(transforms_h.views(), transforms.views())) {
        init_pos.store(transform_h.ptr(), transform_h.stride(1));
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
    // const f32 dt_nd = last_panel_chord / linalg::length(assembly.kinematics()->linear_velocity(0.0f, {0.f, 0.f, 0.f}));
    const f32 dt_nd = last_panel_chord;
    const i64 t_steps = static_cast<i64>(t_final_nd / dt_nd);

    std::cout << "dt_nd: " << dt_nd << "\n";
    std::cout << "t_steps: " << t_steps << "\n";

    // 2. Allocate the wake geometry
    {
        MultiDim<2> wake_panels_2D;
        MultiDim<3> verts_wake_3D;
        MultiDim<3> wake_connectivity_3D;
        for (const auto& [ns, nc] : assembly_wings) {
            wake_panels_2D.push_back({ns, t_steps-1});
            verts_wake_3D.push_back({ns+1, t_steps, 4});
            wake_connectivity_3D.push_back({4, ns, t_steps-1});
        }
        gamma_wake.init(wake_panels_2D);
        verts_wake.init(verts_wake_3D);
        verts_wake_h.init(verts_wake_3D);
        wake_cell_type.init(wake_panels_2D);
        wake_cell_offset.init(wake_panels_2D);
        wake_cell_connectivity.init(wake_connectivity_3D);

        multibody_connectivity(verts_wake_h.views(), wake_cell_type.views(), wake_cell_offset.views(), wake_cell_connectivity.views());
        t_h.init({t_steps-2});
    }

    initialize_structure_data(vars, t_steps, dt_nd);

    verts_wake_h.views()[0].fill(0.0f);

    // Precomputation (only valid in single body case)
    lhs.view().fill(0.f);
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0 , 0);
    solver->factorize(lhs.view());
    backend->wake_shed(verts_wing.views(), verts_wake.views(), 0);
    wing_velocities(assembly.surface_kinematics()[0]->transform_dual(0.0f), colloc_h.views()[0], velocities_h.views()[0], velocities.views()[0]);
    rhs.view().fill(0.f);
    backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
    backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), 0);
    solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
    update_wake_and_gamma(0);

    std::ofstream data_2dof_aero("2dof_aero.txt");
    data_2dof_aero << 1.0f << "\n";

    std::ofstream data_2dof("2dof.txt");
    data_2dof << 1.0f << "\n";

    std::vector<tiny::VtuMesh<f32, i64, i8>> vtu_meshes;
    for (i64 m = 0; m < assembly_wings.size(); m++) {
        const auto& verts_wing_m = verts_wing_h.views()[m];
        const auto& wing_cell_type_m = wing_cell_type.views()[m];
        const auto& wing_cell_offset_m = wing_cell_offset.views()[m];
        const auto& wing_cell_connectivity_m = wing_cell_connectivity.views()[m];
        const auto& verts_wake_m = verts_wake_h.views()[m];
        const auto& wake_cell_type_m = wake_cell_type.views()[m];
        const auto& wake_cell_offset_m = wake_cell_offset.views()[m];
        const auto& wake_cell_connectivity_m = wake_cell_connectivity.views()[m];

        vtu_meshes.push_back(tiny::VtuMesh<f32, i64, i8>{
            std::make_unique<TensorViewAccessor<f32>>(verts_wing_m.slice(All, All, Range{0, 3}).reshape(verts_wing_m.shape(0)*verts_wing_m.shape(1), verts_wing_m.shape(2)-1)),
            std::make_unique<TensorViewAccessor<i64>>(wing_cell_connectivity_m.reshape(wing_cell_connectivity_m.size(), 1)),
            std::make_unique<TensorViewAccessor<i64>>(wing_cell_offset_m.reshape(wing_cell_offset_m.size(), 1)),
            std::make_unique<TensorViewAccessor<i8>>(wing_cell_type_m.reshape(wing_cell_type_m.size(), 1))
        });

        vtu_meshes.push_back(tiny::VtuMesh<f32, i64, i8>{
            std::make_unique<TensorViewAccessor<f32>>(verts_wake_m.slice(All, All, Range{0, 3}).reshape(verts_wake_m.shape(0)*verts_wake_m.shape(1), verts_wake_m.shape(2)-1)),
            std::make_unique<TensorViewAccessor<i64>>(wake_cell_connectivity_m.reshape(wake_cell_connectivity_m.size(), 1)),
            std::make_unique<TensorViewAccessor<i64>>(wake_cell_offset_m.reshape(wake_cell_offset_m.size(), 1)),
            std::make_unique<TensorViewAccessor<i8>>(wake_cell_type_m.reshape(wake_cell_type_m.size(), 1))
        });
    }

    std::unique_ptr<tiny::VtuDataAccessor<f32>> vtu_timesteps = std::make_unique<TensorViewAccessor<f32>>(t_h.view().reshape(t_h.view().size(), 1));

    // Transient simulation loop
    for (i32 i = 0; i < t_steps-2; i++) {
        const f32 t_nd = (f32)i * dt_nd;
        t_h.view()(i) = t_nd;

        #ifdef COUPLED

        data_2dof << t_nd << " " << u_h.view()(0,i) << " " << u_h.view()(1,i) << "\n";

        // Output:
        for (const auto& [wake_d, wake_h] : zip(verts_wake.views(), verts_wake_h.views())) {
            wake_d.slice(All, -1-i, All).to(wake_h.slice(All, -1-i, All));
            wake_d.slice(All, -1-i, All).to(wake_h.slice(All, -2-i, All));
        }

        verts_wing.views()[0].to(verts_wing_h.views()[0]); // for output
        const f32 w_alpha = wing_alpha();
        const f32 w_h_nd = wing_h(1 + vars.a_h);
        ASSERT_NEAR(w_h_nd, u_h.view()(0, i), 5e-6f);
        ASSERT_NEAR(w_alpha, u_h.view()(1, i), 5e-6f);

        for (i64 m = 0; m < assembly_wings.size(); m++) {
            const auto& verts_wake_m = verts_wake_h.views()[m];
            const auto& wake_cell_type_m = wake_cell_type.views()[m];
            const auto& wake_cell_offset_m = wake_cell_offset.views()[m];
            const auto& wake_cell_connectivity_m = wake_cell_connectivity.views()[m];

            const auto verts_slice = verts_wake_m.slice(All, Range{-2-i, verts_wake_m.shape(1)}, Range{0, 3});
            const auto cell_connectivity_slice = wake_cell_connectivity_m.slice(All, All, Range{-1-i, wake_cell_connectivity_m.shape(2)});
            const auto cell_offset_slice = wake_cell_offset_m.slice(All, Range{-1-i, wake_cell_offset_m.shape(1)});
            const auto cell_type_slice = wake_cell_type_m.slice(All, Range{-1-i, wake_cell_type_m.shape(1)});

            body_connectivity(verts_slice, cell_type_slice, cell_offset_slice, cell_connectivity_slice);

            vtu_meshes[2*m+1] = tiny::VtuMesh<f32, i64, i8>{
                std::make_unique<TensorViewAccessor<f32>>(verts_slice.reshape(verts_slice.shape(0)*verts_slice.shape(1), verts_slice.shape(2))),
                std::make_unique<TensorViewAccessor<i64>>(cell_connectivity_slice.reshape(cell_connectivity_slice.size(), 1)),
                std::make_unique<TensorViewAccessor<i64>>(cell_offset_slice.reshape(cell_offset_slice.size(), 1)),
                std::make_unique<TensorViewAccessor<i8>>(cell_type_slice.reshape(cell_type_slice.size(), 1))
            };
        }

        std::string base_it_name = "2dof_" + std::to_string(i);
        tiny::write_pvtu<f32>("./2dof/", base_it_name, {}, {}, vtu_meshes.size());
        for (i64 m = 0; m < vtu_meshes.size(); m++) {
            tiny::write_vtu<f32, i64, i8, f32>("./2dof/" + base_it_name + "/" + base_it_name + "_" + std::to_string(m) + ".vtu", vtu_meshes[m], {}, {});
        }
        
        dF.view().fill(0.f);
        du.view().fill(0.f);
        du_k.view().fill(1.f);
        i32 iteration = 0;
        const i32 max_iter = 50;

        while (backend->blas->norm(du_k.view()) > 1e-7f) {
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

            // Returns dimensionalized values
            const KinematicNode heave([&](const fwd::Float& t) {
                return translation_matrix<fwd::Float>({
                    0.0f,
                    0.0f,
                    // u_h.view()(0, i+1)
                    u_h.view()(0, i+1) + v_h.view()(0, i+1) * t
                });
            });
            const KinematicNode pitch([&](const fwd::Float& t) {
                return rotation_matrix<fwd::Float>(
                    {
                        1.f + vars.a_h,
                        0.0f,
                        0.0f
                    },
                    {
                        0.0f,
                        1.0f,
                        0.0f
                    }, 
                    // u_h.view()(1, i+1)
                    u_h.view()(1, i+1) + v_h.view()(1, i+1) * t
                );
            });

            std::printf("iter: %d | heave: %f | pitch: %f\n", iteration, u_h.view()(0, i+1), u_h.view()(1, i+1));
            
            // Manually compute the transform (update_transforms(assembly, t+dt))
            auto fs = assembly.kinematics()->transform_dual(t_nd+dt_nd);
            auto transform_dual = linalg::mul(linalg::mul(fs, heave.transform_dual(0.0f)), pitch.transform_dual(0.0f));
            auto transform = dual_to_float(transform_dual);
            transform.store(transforms_h.views()[0].ptr(), transforms_h.views()[0].stride(1));
            transforms_h.views()[0].to(transforms.views()[0]);

            // Aero
            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
            backend->wake_shed(verts_wing.views(), verts_wake.views(), i+1);

            wing_velocities(transform_dual, colloc_h.views()[0], velocities_h.views()[0], velocities.views()[0]);
            
            rhs.view().fill(0.f);
            backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
            backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), i+1);
            solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
            update_wake_and_gamma(i+1);

            const linalg::float3 freestream = -assembly.kinematics()->linear_velocity(t_nd+dt_nd, {0.f, 0.f, 0.f});
            backend->forces_unsteady(
                verts_wing.views()[0],
                gamma_wing_delta.views()[0],
                gamma_wing.views()[0],
                gamma_wing_prev.views()[0],
                velocities.views()[0],
                areas_d.views()[0],
                normals_d.views()[0],
                aero_forces.views()[0],
                dt_nd
            );

            const f32 cl = backend->coeff_cl(
                aero_forces.views()[0],
                linalg::normalize(linalg::cross(freestream, {0.f, 1.f, 0.f})), // lift axis
                freestream,
                vars.rho,
                backend->sum(areas_d.views()[0])
            );

            auto ref_pt_4 = linalg::mul(transform, {1 + vars.a_h, 0.0f, 0.0f, 1.0f});
            
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

            F_h.view()(0, i+1) = - cl / (PI_f * vars.mu);
            F_h.view()(1, i+1) = (2.f*cm.y) / (PI_f * vars.mu * pow(vars.r_a, 2));

            dF_h.view()(0) = F_h.view()(0, i+1) - F_h.view()(0, i) - pow(vars.omega / vars.U_a, 2) * du_h.view()(0);
            dF_h.view()(1) = F_h.view()(1, i+1) - F_h.view()(1, i) - 1/pow(vars.U_a, 2) * (torsional_func(u_h.view()(1,i+1)) - torsional_func(u_h.view()(1,i)));
            dF_h.view().to(dF.view());

            backend->blas->axpy(-1.0f, du.view(), du_k.view());
            iteration++;
            if (iteration > max_iter) {
                std::printf("Newton process did not converge\n");
                break;
            }
        } // while loop
        
        std::printf("%d| iters: %d\n", i, iteration);
        #else
        // Classic UVLM:
        update_transforms(assembly, t_nd+dt_nd);
        backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
        backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
        backend->wake_shed(verts_wing.views(), verts_wake.views(), i+1);
        auto transform_dual = assembly.surface_kinematics()[0]->transform_dual(t_nd+dt_nd);
        wing_velocities(transform_dual, colloc_h.views()[0], velocities_h.views()[0], velocities.views()[0]);

        rhs.view().fill(0.f);
        backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
        backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), i+1);
        solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
        update_wake_and_gamma(i+1);

        const linalg::float3 freestream = -assembly.kinematics()->linear_velocity(t_nd+dt_nd, {0.f, 0.f, 0.f});
        backend->forces_unsteady(
            verts_wing.views()[0],
            gamma_wing_delta.views()[0],
            gamma_wing.views()[0],
            gamma_wing_prev.views()[0],
            velocities.views()[0],
            areas_d.views()[0],
            normals_d.views()[0],
            aero_forces.views()[0],
            dt_nd
        );

        const f32 cl = backend->coeff_cl(
            aero_forces.views()[0],
            linalg::normalize(linalg::cross(freestream, {0.f, 1.f, 0.f})), // lift axis
            freestream,
            vars.rho,
            backend->sum(areas_d.views()[0])
        );

        auto ref_pt_4 = linalg::mul(dual_to_float(transform_dual), {1 + vars.a_h, 0.0f, 0.0f, 1.0f});
            
        const linalg::float3 cm = backend->coeff_cm(
            aero_forces.views()[0],
            verts_wing.views()[0],
            {ref_pt_4.x, ref_pt_4.y, ref_pt_4.z},
            freestream,
            vars.rho,
            backend->sum(areas_d.views()[0]),
            backend->mesh_mac(verts_wing.views()[0], areas_d.views()[0])
        );

        data_2dof_aero << t_nd << " " << cl << " " << cm.y <<"\n";
        #endif
    } // simulation loop

    tiny::write_pvd<f32>(".", "2dof", ".pvtu", vtu_timesteps);
}

int main() {
    // vlm::Executor::instance(1);
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_10x5.x"};
    // const std::vector<std::string> meshes = {"../../../../mesh/rectangular_2x2.x"};
    const std::vector<std::string> backends = {"cpu"};

    auto simulations = tiny::make_combination(meshes, backends);

    const f32 flutter_speed = 6.285f;
    const f32 flutter_ratio = 0.2f;
    const f32 t_final_nd = 90.f;

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
    #ifdef COUPLED
    vars.U_a = flutter_ratio * flutter_speed;
    #else
    vars.U_a = 4.0f; // THIS IS NOT ND U_INF
    #endif

    KinematicsTree kinematics_tree;

    #ifdef COUPLED
    auto freestream = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            -t,
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
    #else 

    auto freestream = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            -t,
            0.0f,
            0.0f
        });
    });

    const f32 k = 0.5; // reduced frequency
    const f32 omega = k / vars.b;
    const f32 amplitude = 1.f; // amplitude in degrees
    auto pitch = kinematics_tree.add([=](const fwd::Float& t) {
        return rotation_matrix<fwd::Float>({1 + vars.a_h, 0.0f, 0.0f},{0.0f, 1.0f, 0.0f}, to_radians(amplitude) * fwd::sin(omega * t));
    })->after(freestream);

    for (const auto& [mesh_name, backend_name] : simulations) {
        Assembly assembly(freestream);
        assembly.add(mesh_name, pitch);
        UVLM_2DOF simulation{backend_name, {mesh_name}};
        simulation.run(assembly, vars, t_final_nd);
    }
    #endif
    

    return 0;
}
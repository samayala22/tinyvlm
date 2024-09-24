#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"

#include "tinycombination.hpp"
#include "tinypbar.hpp"

using namespace vlm;

class NewmarkBeta {
    public:
        NewmarkBeta(Memory& memory, const f32 beta = 0.25f, const f32 gamma = 0.5f) : m_memory(memory), m_beta(beta), m_gamma(gamma) {};
        ~NewmarkBeta() = default;

        void init(View<f32, Tensor<2>>& M, View<f32, Tensor<2>>& C, View<f32, Tensor<2>>& K, View<f32, Tensor<1>>& F0, View<f32, Tensor<1>>& u0, View<f32, Tensor<1>>& v0, u32 nb_timesteps);
        void step(View<f32, Tensor<2>>& M, View<f32, Tensor<2>>& C, View<f32, Tensor<2>>& K, View<f32, Tensor<1>>& F_now, View<f32, Tensor<1>>& F_next, const f32 dt, const u32 iteration);
    private:
        std::unique_ptr<BLAS> m_blas;
        Memory& m_memory;
        std::unique_ptr<LUSolver> m_solver;
        const f32 m_beta;
        const f32 m_gamma;
    public:
        Buffer<f32, MemoryLocation::Device, Tensor<2>> K_eff{m_memory}; // effective stiffness
        Buffer<f32, MemoryLocation::Device, Tensor<1>> du{m_memory}; // incremental displacement
        Buffer<f32, MemoryLocation::Device, Tensor<1>> factor{m_memory}; // intermediary vector
        Buffer<f32, MemoryLocation::Device, Tensor<2>> u{m_memory}; // dof x tsteps position history
        Buffer<f32, MemoryLocation::Device, Tensor<2>> v{m_memory}; // dof x tsteps velocity history
        Buffer<f32, MemoryLocation::Device, Tensor<2>> a{m_memory}; // dof x tsteps acceleration history
};

void NewmarkBeta::init(View<f32, Tensor<2>>& M, View<f32, Tensor<2>>& C, View<f32, Tensor<2>>& K, View<f32, Tensor<1>>& F0, View<f32, Tensor<1>>& u0, View<f32, Tensor<1>>& v0, u32 nb_timesteps) {
    assert(M.layout.shape(0) == C.layout.shape(0));
    assert(M.layout.shape(1) == C.layout.shape(1));
    assert(C.layout.shape(0) == K.layout.shape(0));
    assert(C.layout.shape(1) == K.layout.shape(1));

    const Tensor<2> time_series{{M.layout.shape(0), nb_timesteps}}; // dofs x timesteps

    K_eff.dealloc();
    du.dealloc();
    factor.dealloc();
    u.dealloc();
    v.dealloc();
    a.dealloc();

    K_eff.alloc(K.layout);
    du.alloc(F0.layout);
    factor.alloc(F0.layout);
    u.alloc(time_series);
    v.alloc(time_series);
    a.alloc(time_series);

    m_memory.fill_f32(MemoryLocation::Device, u.d_view().ptr, 0.0f, u.size());
    m_memory.fill_f32(MemoryLocation::Device, v.d_view().ptr, 0.0f, v.size());
    m_memory.fill_f32(MemoryLocation::Device, a.d_view().ptr, 0.0f, a.size());

    View<f32, Tensor<1>> a0_col0 = a.d_view().layout.slice(a.d_view().ptr, all, 0);
    m_memory.copy(MemoryTransfer::DeviceToDevice, a0_col0.ptr, F0.ptr, F0.size_bytes());
    m_blas->gemv(-1.0f, C, v0, 1.0f, a0_col0);
    m_blas->gemv(-1.0f, K, u0, 1.0f, a0_col0);
    m_solver->factorize(M);
    m_solver->solve(M, a0_col0);
}

void NewmarkBeta::step(View<f32, Tensor<2>>& M, View<f32, Tensor<2>>& C, View<f32, Tensor<2>>& K, View<f32, Tensor<1>>& F_now, View<f32, Tensor<1>>& F_next, const f32 dt, const u32 iteration) {
    const f32 x1 = m_gamma / (m_beta * dt);
    const f32 x0 = 1 / (m_beta * dt*dt);
    const f32 xd0 = 1 / (m_beta * dt);
    const f32 xd1 = m_gamma / m_beta;
    const f32 xdd0 = 1/(2*m_beta);
    const f32 xdd1 = - dt * (1 - m_gamma / (2*m_beta));

    // K_eff = K + a0 * M + a1 * C
    m_memory.copy(MemoryTransfer::DeviceToDevice, K_eff.d_view().ptr, K.ptr, K.size_bytes());
    m_blas->axpy(x0, M, K_eff.d_view());
    m_blas->axpy(x1, C, K_eff.d_view());

    // F_eff = (F[i+1]-F[i]) + M @ (xd0 * v[i] + xdd0 * a[i]) + C @ (xd1 * v[i] + xdd1 * a[i])
    m_memory.fill_f32(MemoryLocation::Device, du.d_view().ptr, 0.0f, du.size());
    m_blas->axpy(1.0f, F_next, du.d_view());
    m_blas->axpy(-1.0f, F_now, du.d_view());

    View<f32, Tensor<1>> a_i = a.d_view().layout.slice(a.d_view().ptr, all, iteration);
    View<f32, Tensor<1>> v_i = v.d_view().layout.slice(v.d_view().ptr, all, iteration);
    View<f32, Tensor<1>> u_i = u.d_view().layout.slice(u.d_view().ptr, all, iteration);

    m_memory.fill_f32(MemoryLocation::Device, factor.d_view().ptr, 0.0f, factor.size());
    m_blas->axpy(xd0, v_i, factor.d_view());
    m_blas->axpy(xdd0, a_i, factor.d_view());
    m_blas->gemv(1.0f, M, factor.d_view(), 1.0f, du.d_view());

    m_memory.fill_f32(MemoryLocation::Device, factor.d_view().ptr, 0.0f, factor.size());
    m_blas->axpy(xd1, v_i, factor.d_view());
    m_blas->axpy(xdd1, a_i, factor.d_view());
    m_blas->gemv(1.0f, C, factor.d_view(), 1.0f, du.d_view());

    m_solver->factorize(K_eff.d_view());
    m_solver->solve(K_eff.d_view(), du.d_view());

    View<f32, Tensor<1>> a_ip1 = a.d_view().layout.slice(a.d_view().ptr, all, iteration+1);
    View<f32, Tensor<1>> v_ip1 = v.d_view().layout.slice(v.d_view().ptr, all, iteration+1);
    View<f32, Tensor<1>> u_ip1 = u.d_view().layout.slice(u.d_view().ptr, all, iteration+1);

    // u[i+1] = u[i] + du
    // v[i+1] = v[i] + x1 * du - xd1 * v[i] - xdd1 * a[i]
    // a[i+1] = a[i] + x0 * du - xd0 * v[i] - xdd0 * a[i]
    // Fused kernel (only for host)
    for (u32 i = 0; i < u.d_view().layout.shape(0); i++) {
        u_ip1[i] = u_i[i] + du.d_view()[i];
        v_ip1[i] = v_i[i] + x1 * du.d_view()[i] - xd1 * v_i[i] - xdd1 * a_i[i];
        a_ip1[i] = a_i[i] + x0 * du.d_view()[i] - xd0 * v_i[i] - xdd0 * a_i[i];
    }
}

class UVLM_2DOF final: public Simulation {
    public:
        UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~UVLM_2DOF() = default;
        void run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::alias::float4x4>& initial_pose, f32 t_final);
    
        // Mesh
        Buffer<f32, MemoryLocation::HostDevice, MultiSurface> verts_wing_pos{*backend->memory}; // (nc+1)*(ns+1)*3
        Buffer<f32, MemoryLocation::Host, MultiSurface> colloc_pos{*backend->memory}; // (nc)*(ns)*3

        // Data
        Buffer<f32, MemoryLocation::Device, Matrix<MatrixLayout::ColMajor>> lhs{*backend->memory}; // (ns*nc)^2
        Buffer<f32, MemoryLocation::Device, MultiSurface> rhs{*backend->memory}; // ns*nc
        Buffer<f32, MemoryLocation::HostDevice, MultiSurface> gamma_wing{*backend->memory}; // nc*ns
        Buffer<f32, MemoryLocation::Device, MultiSurface> gamma_wake{*backend->memory}; // nw*ns
        Buffer<f32, MemoryLocation::Device, MultiSurface> gamma_wing_prev{*backend->memory}; // nc*ns
        Buffer<f32, MemoryLocation::Device, MultiSurface> gamma_wing_delta{*backend->memory}; // nc*ns
        Buffer<f32, MemoryLocation::HostDevice, MultiSurface> velocities{*backend->memory}; // ns*nc*3
        Buffer<f32, MemoryLocation::HostDevice, Tensor<3>> transforms{*backend->memory}; // 4*4*nb_meshes
        
        // Simulation boilerplate
        std::vector<f32> vec_t; // timesteps
        std::vector<f32> local_dt; // per mesh dt (pre reduction)
        std::vector<u32> condition0;

        // Structure 
        Buffer<f32, MemoryLocation::Device, Tensor<2>> M{*backend->memory}; // dof x dof mass matrix
        Buffer<f32, MemoryLocation::Device, Tensor<2>> C{*backend->memory}; // dof x dof damping matrix
        Buffer<f32, MemoryLocation::Device, Tensor<2>> K{*backend->memory}; // dof x dof stiffness matrix
        Buffer<f32, MemoryLocation::Device, Tensor<2>> F{*backend->memory}; // dof x timesteps

        NewmarkBeta integrator{*backend->memory};
    private:
        void alloc_buffers();
};

UVLM_2DOF::UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    alloc_buffers();
}

void UVLM_2DOF::alloc_buffers() {
    const u64 n = wing_panels.back().offset + wing_panels.back().size();

    // Mesh
    verts_wing_pos.alloc(MultiSurface{wing_vertices, 4});
    colloc_pos.alloc(MultiSurface{wing_panels, 3});

    // Data
    lhs.alloc(Matrix<MatrixLayout::ColMajor>{n, n, n});
    rhs.alloc(MultiSurface{wing_panels, 1});
    gamma_wing.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_prev.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_delta.alloc(MultiSurface{wing_panels, 1});
    velocities.alloc(MultiSurface{wing_panels, 3});
    transforms.alloc(Tensor<3>({4,4,nb_meshes}));

    local_dt.resize(nb_meshes);
    condition0.resize(nb_meshes);

    backend->lu_allocate(lhs.d_view()); // TODO: maybe move this ?

    const u32 dof = 2;
    const Tensor<2> structure_layout{{dof, dof}};
    M.alloc(structure_layout);
    C.alloc(structure_layout);
    K.alloc(structure_layout);
}

void UVLM_2DOF::run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::alias::float4x4>& initial_pose, f32 t_final) {
    mesh.verts_wing_init.to_device(); // raw mesh vertices from file
    for (u64 m = 0; m < kinematics.size(); m++) {
        initial_pose[m].store(transforms.h_view().ptr + m*transforms.h_view().layout.stride(2), transforms.h_view().layout.stride(1));
    }
    transforms.to_device();
    backend->displace_wing(transforms.d_view(), verts_wing_pos.d_view(), mesh.verts_wing_init.d_view());
    backend->memory->copy(MemoryTransfer::DeviceToHost, colloc_pos.h_view().ptr, mesh.colloc.d_view().ptr, mesh.colloc.d_view().size_bytes());
    backend->memory->copy(MemoryTransfer::DeviceToDevice, mesh.verts_wing.d_view().ptr, verts_wing_pos.d_view().ptr, mesh.verts_wing.d_view().size_bytes());
    backend->memory->fill_f32(MemoryLocation::Device, lhs.d_view().ptr, 0.f, lhs.d_view().size());
    backend->mesh_metrics(0.0f, mesh.verts_wing.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.area.d_view());

    // 1.  Compute the dynamic time steps for the simulation
    verts_wing_pos.to_host();
    vec_t.clear();
    vec_t.push_back(0.0f);
    for (f32 t = 0.0f; t < t_final;) {
        // parallel for
        for (u64 m = 0; m < nb_meshes; m++) {
            local_dt[m] = std::numeric_limits<f32>::max();
            const auto local_transform = kinematics[m].displacement(t);
            const f32* local_verts = verts_wing_pos.h_view().ptr + verts_wing_pos.h_view().layout.offset(m);
            const u64 pre_trailing_begin = (verts_wing_pos.h_view().layout.nc(m)-2)*verts_wing_pos.h_view().layout.ns(m);
            const u64 trailing_begin = (verts_wing_pos.h_view().layout.nc(m)-1)*verts_wing_pos.h_view().layout.ns(m);
            // parallel for
            for (u64 j = 0; j < verts_wing_pos.h_view().layout.ns(m); j++) {
                const f32 local_chord_dx = local_verts[0*verts_wing_pos.h_view().layout.stride() + pre_trailing_begin + j] - local_verts[0*verts_wing_pos.h_view().layout.stride() + trailing_begin + j];
                const f32 local_chord_dy = local_verts[1*verts_wing_pos.h_view().layout.stride() + pre_trailing_begin + j] - local_verts[1*verts_wing_pos.h_view().layout.stride() + trailing_begin + j];
                const f32 local_chord_dz = local_verts[2*verts_wing_pos.h_view().layout.stride() + pre_trailing_begin + j] - local_verts[2*verts_wing_pos.h_view().layout.stride() + trailing_begin + j];
                const f32 local_chord = std::sqrt(local_chord_dx*local_chord_dx + local_chord_dy*local_chord_dy + local_chord_dz*local_chord_dz);
                local_dt[m] = std::min(local_dt[m], local_chord / kinematics[m].velocity_magnitude(local_transform, {
                    local_verts[0*verts_wing_pos.h_view().layout.stride() + trailing_begin + j],
                    local_verts[1*verts_wing_pos.h_view().layout.stride() + trailing_begin + j],
                    local_verts[2*verts_wing_pos.h_view().layout.stride() + trailing_begin + j],
                    1.0f
                }));
            }
        }
        // parallel reduce
        f32 dt = std::numeric_limits<f32>::max();
        for (u64 m = 0; m < kinematics.size(); m++) {
            dt = std::min(dt, local_dt[m]);
        }
        t += std::min(dt, t_final - t);
        vec_t.push_back(t);
    }

    // 2. Allocate the wake geometry
    {
        wake_panels.clear();
        wake_vertices.clear();

        gamma_wake.dealloc();
        mesh.verts_wake.dealloc();
        
        const u64 nw = vec_t.size()-1;
        u64 off_wake_p = 0;
        u64 off_wake_v = 0;
        for (u32 i = 0; i < wing_panels.size(); i++) {
            wake_panels.emplace_back(SurfaceDims{nw, wing_panels[i].ns, off_wake_p});
            wake_vertices.emplace_back(SurfaceDims{nw+1, wing_vertices[i].ns, off_wake_v});
            off_wake_p += wake_panels.back().size();
            off_wake_v += wake_vertices.back().size();
        }
        gamma_wake.alloc(MultiSurface{wake_panels, 1});
        mesh.alloc_wake(wake_vertices);
    }

    // 3. Precompute constant values for the transient simulation
    backend->lhs_assemble(lhs.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), condition0 , 0);
    backend->lu_factor(lhs.d_view());

    // 4. Transient simulation loop
    // for (u32 i = 0; i < vec_t.size()-1; i++) {
    for (const i32 i : tiny::pbar(0, (i32)vec_t.size()-1)) {
        const f32 t = vec_t[i];
        const f32 dt = vec_t[i+1] - t;

        const linalg::alias::float3 freestream = -kinematics[0].velocity(kinematics[0].displacement(t,1), {0.f, 0.f, 0.f, 1.0f});

        backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), i);

        // parallel for
        for (u64 m = 0; m < nb_meshes; m++) {
            const auto local_transform = kinematics[m].displacement(t);
            const f32* local_colloc = colloc_pos.h_view().ptr + colloc_pos.h_view().layout.offset(m);
            f32* local_velocities = velocities.h_view().ptr + velocities.h_view().layout.offset(m);
            
            // parallel for
            for (u64 idx = 0; idx < colloc_pos.h_view().layout.surface(m).size(); idx++) {
                auto local_velocity = -kinematics[m].velocity(local_transform, {
                    local_colloc[0*colloc_pos.h_view().layout.stride() + idx],
                    local_colloc[1*colloc_pos.h_view().layout.stride() + idx],
                    local_colloc[2*colloc_pos.h_view().layout.stride() + idx],
                    1.0f
                });
                local_velocities[0*velocities.h_view().layout.stride() + idx] = local_velocity.x;
                local_velocities[1*velocities.h_view().layout.stride() + idx] = local_velocity.y;
                local_velocities[2*velocities.h_view().layout.stride() + idx] = local_velocity.z;
            }
        }

        velocities.to_device();
        backend->memory->fill_f32(MemoryLocation::Device, rhs.d_view().ptr, 0.f, rhs.d_view().size());
        backend->rhs_assemble_velocities(rhs.d_view(), mesh.normals.d_view(), velocities.d_view());
        backend->rhs_assemble_wake_influence(rhs.d_view(), gamma_wake.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.verts_wake.d_view(), i);
        backend->lu_solve(lhs.d_view(), rhs.d_view(), gamma_wing.d_view());
        backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());
        
        // skip cl computation for the first iteration
        if (i > 0) {
            const f32 cl_unsteady = backend->coeff_unsteady_cl_multi(mesh.verts_wing.d_view(), gamma_wing_delta.d_view(), gamma_wing.d_view(), gamma_wing_prev.d_view(), velocities.d_view(), mesh.area.d_view(), mesh.normals.d_view(), freestream, dt);
            // if (i == vec_t.size()-2) std::printf("t: %f, CL: %f\n", t, cl_unsteady);
            // std::printf("t: %f, CL: %f\n", t, cl_unsteady);
            // mesh.verts_wing.to_host();
            // cl_data << t << " " << *(mesh.verts_wing.h_view().ptr + 2*mesh.verts_wing.h_view().layout.stride()) << " " << cl_unsteady << " " << 0.f << "\n";
        }

        // backend->displace_wake_rollup(wake_rollup.d_view(), mesh.verts_wake.d_view(), mesh.verts_wing.d_view(), gamma_wing.d_view(), gamma_wake.d_view(), dt, i);
        backend->gamma_shed(gamma_wing.d_view(), gamma_wing_prev.d_view(), gamma_wake.d_view(), i);
        
        // parallel for
        for (u64 m = 0; m < nb_meshes; m++) {
            const auto local_transform = dual_to_float(kinematics[m].displacement(t+dt));
            local_transform.store(transforms.h_view().ptr + m*transforms.h_view().layout.stride(2), transforms.h_view().layout.stride(1));
        }
        transforms.to_device();
        backend->displace_wing(transforms.d_view(), mesh.verts_wing.d_view(), verts_wing_pos.d_view());
        backend->mesh_metrics(0.0f, mesh.verts_wing.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.area.d_view());
    }
}

int main() {
    const u64 ni = 20;
    const u64 nj = 5;
    // vlm::Executor::instance(1);
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_" + std::to_string(ni) + "x" + std::to_string(nj) + ".x"};
    const std::vector<std::string> backends = {"cpu"};

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 10.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 0.5; // reduced frequency
    const f32 omega = k * 2.0f * u_inf / (2*b);
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods

    Kinematics kinematics{};

    const f32 initial_angle = 0.0f;

    const auto initial_pose = rotation_matrix(
        linalg::alias::float3{0.0f, 0.0f, 0.0f}, // take into account quarter chord panel offset
        linalg::alias::float3{0.0f, 1.0f, 0.0f},
        to_radians(initial_angle)
    );
    
    // Sudden acceleration
    const f32 alpha = to_radians(5.0f);
    kinematics.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            -u_inf*std::cos(alpha)*t,
            0.0f,
            -u_inf*std::sin(alpha)*t
        });
    });

    for (const auto& [mesh_name, backend_name] : solvers) {
        UVLM simulation{backend_name, {mesh_name}};
        simulation.run({kinematics}, {initial_pose}, t_final);
    }
    return 0;
}
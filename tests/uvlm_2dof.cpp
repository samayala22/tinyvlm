#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"
#include "vlm_integrator.hpp"

#include "tinycombination.hpp"
#include "tinypbar.hpp"

#include <tuple>
#include <fstream>

using namespace vlm;

constexpr i32 DOF = 2;
// #define DEBUG_DISPLACEMENT_DATA

class UVLM_2DOF final: public Simulation {
    public:
        UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~UVLM_2DOF() = default;
        void run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::float4x4>& initial_pose, f32 t_final);
    
        // Mesh (initial position)
        // Buffer<f32, Location::HostDevice, MultiSurface> verts_wing_pos{*backend->memory}; // (nc+1)*(ns+1)*3
        // Buffer<f32, Location::Host, MultiSurface> colloc_h{*backend->memory}; // (nc)*(ns)*3

        // Aero
        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        // Buffer<f32, Location::HostDevice, MultiSurface> gamma_wing{*backend->memory}; // nc*ns
        // Buffer<f32, Location::Device, MultiSurface> gamma_wake{*backend->memory}; // nw*ns
        // Buffer<f32, Location::Device, MultiSurface> gamma_wing_prev{*backend->memory}; // nc*ns
        // Buffer<f32, Location::Device, MultiSurface> gamma_wing_delta{*backend->memory}; // nc*ns
        // Buffer<f32, Location::HostDevice, MultiSurface> velocities{*backend->memory}; // ns*nc*3
        Tensor3D<Location::Host> transforms_h{backend->memory.get()};
        Tensor3D<Location::Device> transforms{backend->memory.get()}; // 4*4*nb_meshes
        Tensor3D<Location::Device> panel_forces{backend->memory.get()}; // panels x 3 x timesteps

        // Boilerplate (todo: delete this ugly stuff)
        std::vector<i32> condition0;
        std::vector<linalg::float4x4> body_frames; // todo: move this somewhere else

        // Structure
        Tensor2D<Location::Host> M_h{backend->memory.get()};
        Tensor2D<Location::Host> C_h{backend->memory.get()};
        Tensor2D<Location::Host> K_h{backend->memory.get()};
        Tensor2D<Location::Host> u_h{backend->memory.get()};
        Tensor2D<Location::Host> v_h{backend->memory.get()};
        Tensor2D<Location::Host> a_h{backend->memory.get()};
        Tensor1D<Location::Host> t_h{backend->memory.get()};

        Tensor2D<Location::Device> M_d{backend->memory.get()};
        Tensor2D<Location::Device> C_d{backend->memory.get()};
        Tensor2D<Location::Device> K_d{backend->memory.get()};
        Tensor2D<Location::Device> u_d{backend->memory.get()}; // dof x tsteps
        Tensor2D<Location::Device> v_d{backend->memory.get()}; // dof x tsteps
        Tensor2D<Location::Device> a_d{backend->memory.get()}; // dof x tsteps
        Tensor1D<Location::Device> du{backend->memory.get()}; // dof
        Tensor1D<Location::Device> dv{backend->memory.get()}; // dof
        Tensor1D<Location::Device> da{backend->memory.get()}; // dof

        NewmarkBeta integrator{backend.get()};

        std::unique_ptr<LU> solver;
    private:
        void alloc_buffers();
};

UVLM_2DOF::UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    alloc_buffers();
    solver = backend->create_lu_solver();
}

void UVLM_2DOF::alloc_buffers() {
    // const i64 n = wing_panels.back().offset + wing_panels.back().size();

    // // Mesh
    // verts_wing_pos.alloc(MultiSurface{wing_vertices, 4});
    // colloc_h.alloc(MultiSurface{wing_panels, 3});
    // MultiDim<3> panels_3D;
    // for (auto& [ns, nc] : assembly_wings) {
    //     panels_3D.push_back({ns, nc, 3});
    // }
    // normals_d.init(panels_3D);
    // colloc_d.init(panels_3D);

    // // Data
    // lhs.init({n, n});
    // rhs.init({n});
    // gamma_wing.alloc(MultiSurface{wing_panels, 1});
    // gamma_wing_prev.alloc(MultiSurface{wing_panels, 1});
    // gamma_wing_delta.alloc(MultiSurface{wing_panels, 1});
    // velocities.alloc(MultiSurface{wing_panels, 3});
    // transforms_h.init({4,4,nb_meshes});
    // transforms.init({4,4,nb_meshes});
    // solver->init(lhs.view());

    // condition0.resize(nb_meshes);
    // body_frames.resize(nb_meshes, linalg::identity);

    // Host tensors
    M_h.init({DOF,DOF});
    C_h.init({DOF,DOF});
    K_h.init({DOF,DOF});

    // Device tensors
    M_d.init({DOF,DOF});
    C_d.init({DOF,DOF});
    K_d.init({DOF,DOF});
    du.init({DOF});
    dv.init({DOF});
    da.init({DOF});
}

template<typename T>
void dump_buffer(std::ofstream& stream, T* start, i32 size) {
    for (T* it = start; it != start + size; it++) {
        stream << *it << " ";
    }
    stream << "\n";
}

void UVLM_2DOF::run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::float4x4>& initial_pose, f32 t_final) {
    // mesh.verts_wing_init.to_device(); // raw mesh vertices from file
    // for (i64 m = 0; m < kinematics.size(); m++) {
    //     initial_pose[m].store(transforms_h.view().ptr() + transforms_h.view().offset({0, 0, m}), transforms_h.view().stride(1));
    // }
    // transforms_h.view().to(transforms.view());
    // backend->displace_wing(transforms.view(), verts_wing_pos.d_view(), mesh.verts_wing_init.d_view());
    // for (i32 body = 0; body < nb_meshes; body++) {
    //     backend->memory->copy(Location::Host, colloc_h.h_view().ptr + colloc_h.h_view().layout.offset(body), 1, Location::Device, colloc_d.views()[body].ptr(), 1, sizeof(f32), colloc_d.views()[body].size());
    // }
    // backend->memory->copy(Location::Device, mesh.verts_wing.d_view().ptr, 1, Location::Device, verts_wing_pos.d_view().ptr, 1, sizeof(f32), verts_wing_pos.d_view().size());
    // lhs.view().fill(0.f);
    // backend->mesh_metrics(0.0f, mesh.verts_wing.d_view(), colloc_d.views(), normals_d.views(), mesh.area.d_view());

    // 1. Compute static time step for the simulation
    // auto first_wing = mesh_verts_wing.view().slice(Range{0, assembly.size(0)}, All});
    // const f32 local_chord
    const auto freestream_transform = kinematics[0].displacement(0.0f);

    // auto verts_first_wing = verts_wing.view().slice(Range{assembly_wing.begin(0), assembly_wing.end(0), All).reshape(assembly.shape(i), 3)
    // const f32 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    // const f32 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 0);
    // const f32 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 0);
    // const f32 local_chord = std::sqrt(dx*dx + dy*dy + dz*dz);

    // const f32 dx = verts_wing.views(0)
    // verts_wing_pos.to_host();
    // vec_t.clear();
    // vec_t.push_back(0.0f);
    // for (f32 t = 0.0f; t < t_final;) {
    //     // parallel for
    //     for (i64 m = 0; m < nb_meshes; m++) {
    //         local_dt[m] = std::numeric_limits<f32>::max();
    //         const auto local_transform = kinematics[m].displacement(t);
    //         const f32* local_verts = verts_wing_pos.h_view().ptr + verts_wing_pos.h_view().layout.offset(m);
    //         const i64 pre_trailing_begin = (verts_wing_pos.h_view().layout.nc(m)-2)*verts_wing_pos.h_view().layout.ns(m);
    //         const i64 trailing_begin = (verts_wing_pos.h_view().layout.nc(m)-1)*verts_wing_pos.h_view().layout.ns(m);
    //         // parallel for
    //         for (i64 j = 0; j < verts_wing_pos.h_view().layout.ns(m); j++) {
    //             const f32 local_chord_dx = local_verts[0*verts_wing_pos.h_view().layout.stride() + pre_trailing_begin + j] - local_verts[0*verts_wing_pos.h_view().layout.stride() + trailing_begin + j];
    //             const f32 local_chord_dy = local_verts[1*verts_wing_pos.h_view().layout.stride() + pre_trailing_begin + j] - local_verts[1*verts_wing_pos.h_view().layout.stride() + trailing_begin + j];
    //             const f32 local_chord_dz = local_verts[2*verts_wing_pos.h_view().layout.stride() + pre_trailing_begin + j] - local_verts[2*verts_wing_pos.h_view().layout.stride() + trailing_begin + j];
    //             const f32 local_chord = std::sqrt(local_chord_dx*local_chord_dx + local_chord_dy*local_chord_dy + local_chord_dz*local_chord_dz);
    //             local_dt[m] = std::min(local_dt[m], local_chord / kinematics[m].velocity_magnitude(local_transform, {
    //                 local_verts[0*verts_wing_pos.h_view().layout.stride() + trailing_begin + j],
    //                 local_verts[1*verts_wing_pos.h_view().layout.stride() + trailing_begin + j],
    //                 local_verts[2*verts_wing_pos.h_view().layout.stride() + trailing_begin + j],
    //                 1.0f
    //             }));
    //         }
    //     }
    //     // parallel reduce
    //     f32 dt = std::numeric_limits<f32>::max();
    //     for (i64 m = 0; m < kinematics.size(); m++) {
    //         dt = std::min(dt, local_dt[m]);
    //     }
    //     t += std::min(dt, t_final - t);
    //     vec_t.push_back(t);
    // }

    // // 2. Allocate the wake geometry
    // {
    //     wake_panels.clear();
    //     wake_vertices.clear();

    //     gamma_wake.dealloc();
    //     mesh.verts_wake.dealloc();
        
    //     const i64 nw = vec_t.size()-1;
    //     i64 off_wake_p = 0;
    //     i64 off_wake_v = 0;
    //     for (i32 i = 0; i < wing_panels.size(); i++) {
    //         wake_panels.emplace_back(SurfaceDims{nw, wing_panels[i].ns, off_wake_p});
    //         wake_vertices.emplace_back(SurfaceDims{nw+1, wing_vertices[i].ns, off_wake_v});
    //         off_wake_p += wake_panels.back().size();
    //         off_wake_v += wake_vertices.back().size();
    //     }
    //     gamma_wake.alloc(MultiSurface{wake_panels, 1});
    //     mesh.alloc_wake(wake_vertices);
    // }

    // #ifdef DEBUG_DISPLACEMENT_DATA
    // std::ofstream wing_data("wing_data.txt");
    // std::ofstream wake_data("wake_data.txt");

    // wing_data << mesh.verts_wing.d_view().layout.nc(0) << " " << mesh.verts_wing.d_view().layout.ns(0) << "\n";
    // wing_data << vec_t.size() - 1 << "\n\n";

    // wake_data << mesh.verts_wake.d_view().layout.ns(0) << "\n";
    // wake_data << vec_t.size() - 1 << "\n\n";
    // #endif

    // // 3. Precompute constant values for the transient simulation
    // backend->lhs_assemble(lhs.view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), condition0 , 0);
    // solver->factorize(lhs.view());

    // // WARNING: this only works for single body and 2D like movements
    // auto h_alpha = [&]() -> std::tuple<f32, f32> {
    //     // auto inv_transform = linalg::inverse(body_frames[0]);
    //     linalg::float4x4 inv_transform = linalg::identity;
    //     auto& verts = mesh.verts_wing.h_view();
    //     i32 p0_idx = 0; // first vertex on chord
    //     i32 p1_idx = (verts.layout.nc(0)-1) * verts.layout.ns(0); // last vertex on chord
    //     linalg::float4 p0_global = {verts[0*verts.layout.stride() + p0_idx], verts[1*verts.layout.stride() + p0_idx], verts[2*verts.layout.stride() + p0_idx], 1.0f};
    //     linalg::float4 p1_global = {verts[0*verts.layout.stride() + p1_idx], verts[1*verts.layout.stride() + p1_idx], verts[2*verts.layout.stride() + p1_idx], 1.0f};
    //     auto p0 = linalg::mul(inv_transform, p0_global);
    //     auto p1 = linalg::mul(inv_transform, p1_global);
    //     f32 x0 = 0.25f; // point at quarter chord
    //     f32 t = (x0 - p0.x) / (p1.x - p0.x);
    //     f32 z0 = p0.z + t * (p1.z - p0.z);

    //     return std::tuple<f32, f32>(p0.z, std::atan2(p1.z - p0.z, p1.x - p0.x));
    // };

    // panel_forces.init({wing_panels[0].size(), 3, static_cast<i64>(vec_t.size())}); // uuh ??
    // F.init({DOF, static_cast<i64>(vec_t.size()-1)});
    // F.view().fill(0.0f);
    // auto F0 = F.view().slice(All, 0);
    // // integrator.init(M.d_view(), C.d_view(), K.d_view(), F0, u0.d_view(), v0.d_view(), vec_t.size()-1);

    // // 4. Transient simulation loop
    // for (i32 i = 0; i < vec_t.size()-1; i++) {
    //     const auto [h, alpha] = h_alpha();
    //     std::printf("t: %f, h: %f, alpha: %f\n", vec_t[i], h, to_degrees(alpha));
    //     const f32 t = vec_t[i];
    //     const f32 dt = vec_t[i+1] - t;
        
    //     auto panel_forces_ip0 = panel_forces.view().slice(All, All, i);
    //     auto panel_forces_ip1 = panel_forces.view().slice(All, All, i+1);

    //     // Aero 0
    //     {

    //     const linalg::float3 freestream = -kinematics[0].velocity(kinematics[0].displacement(t,1), {0.f, 0.f, 0.f, 1.0f});

    //     backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), i);

    //     // parallel for
    //     for (i64 m = 0; m < nb_meshes; m++) {
    //         const auto local_transform = kinematics[m].displacement(t);
    //         const f32* local_colloc = colloc_h.h_view().ptr + colloc_h.h_view().layout.offset(m);
    //         f32* local_velocities = velocities.h_view().ptr + velocities.h_view().layout.offset(m);
            
    //         // parallel for
    //         for (i64 idx = 0; idx < colloc_h.h_view().layout.surface(m).size(); idx++) {
    //             auto local_velocity = -kinematics[m].velocity(local_transform, {
    //                 local_colloc[0*colloc_h.h_view().layout.stride() + idx],
    //                 local_colloc[1*colloc_h.h_view().layout.stride() + idx],
    //                 local_colloc[2*colloc_h.h_view().layout.stride() + idx],
    //                 1.0f
    //             });
    //             local_velocities[0*velocities.h_view().layout.stride() + idx] = local_velocity.x;
    //             local_velocities[1*velocities.h_view().layout.stride() + idx] = local_velocity.y;
    //             local_velocities[2*velocities.h_view().layout.stride() + idx] = local_velocity.z;
    //         }
    //     }

    //     velocities.to_device();
    //     rhs.view().fill(0.f);
    //     backend->rhs_assemble_velocities(rhs.view(), mesh.normals.d_view(), velocities.d_view());
    //     backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.verts_wake.d_view(), i);
    //     solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
    //     backend->memory->copy(Location::Device, gamma_wing.d_view().ptr, 1, Location::Device, rhs.view().ptr(), 1, sizeof(f32), rhs.view().size());
    //     backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());
        
    //     if (i > 0) {
    //         // backend->coeff_unsteady_cl_multi_forces(mesh.verts_wing.d_view(), gamma_wing_delta.d_view(), gamma_wing.d_view(), gamma_wing_prev.d_view(), velocities.d_view(), mesh.area.d_view(), mesh.normals.d_view(), panel_forces_ip0, freestream, dt);
    //     }

    //     backend->gamma_shed(gamma_wing.d_view(), gamma_wing_prev.d_view(), gamma_wake.d_view(), i);
    //     }

    //     // Move mesh
    //     {
    //         // parallel for
    //         for (i64 m = 0; m < nb_meshes; m++) {
    //             const auto local_transform = dual_to_float(kinematics[m].displacement(t+dt));
    //             body_frames[m] = local_transform;
    //             local_transform.store(transforms_h.view().ptr() + transforms_h.view().offset({0, 0, m}), transforms_h.view().stride(1));
    //         }
    //         transforms_h.view().to(transforms.view());
    //         backend->displace_wing(transforms.view(), mesh.verts_wing.d_view(), verts_wing_pos.d_view());
    //         backend->mesh_metrics(0.0f, mesh.verts_wing.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.area.d_view());
    //     }

    //     // // Aero 1
    //     // {
    //     //     const linalg::float3 freestream = -kinematics[0].velocity(kinematics[0].displacement(t+dt,1), {0.f, 0.f, 0.f, 1.0f});
    //     //     backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), i+1);

    //     //     // parallel for
    //     //     for (i64 m = 0; m < nb_meshes; m++) {
    //     //         const auto local_transform = kinematics[m].displacement(t+dt);
    //     //         const f32* local_colloc = colloc_h.h_view().ptr + colloc_h.h_view().layout.offset(m);
    //     //         f32* local_velocities = velocities.h_view().ptr + velocities.h_view().layout.offset(m);
                
    //     //         // parallel for
    //     //         for (i64 idx = 0; idx < colloc_h.h_view().layout.surface(m).size(); idx++) {
    //     //             auto local_velocity = -kinematics[m].velocity(local_transform, {
    //     //                 local_colloc[0*colloc_h.h_view().layout.stride() + idx],
    //     //                 local_colloc[1*colloc_h.h_view().layout.stride() + idx],
    //     //                 local_colloc[2*colloc_h.h_view().layout.stride() + idx],
    //     //                 1.0f
    //     //             });
    //     //             local_velocities[0*velocities.h_view().layout.stride() + idx] = local_velocity.x;
    //     //             local_velocities[1*velocities.h_view().layout.stride() + idx] = local_velocity.y;
    //     //             local_velocities[2*velocities.h_view().layout.stride() + idx] = local_velocity.z;
    //     //         }
    //     //     }

    //     //     velocities.to_device();
    //     //     backend->memory->fill(Location::Device, rhs.d_view().ptr, 0.f, rhs.d_view().size());
    //     //     backend->rhs_assemble_velocities(rhs.d_view(), mesh.normals.d_view(), velocities.d_view());
    //     //     backend->rhs_assemble_wake_influence(rhs.d_view(), gamma_wake.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.verts_wake.d_view(), i+1);
    //     //     backend->lu_solve(lhs.d_view(), rhs.d_view(), gamma_wing.d_view());
    //     //     backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());
            
    //     //     // backend->coeff_unsteady_cl_multi_forces(mesh.verts_wing.d_view(), gamma_wing_delta.d_view(), gamma_wing.d_view(), gamma_wing_prev.d_view(), velocities.d_view(), mesh.area.d_view(), mesh.normals.d_view(), panel_forces_ip1, freestream, dt);

    //     //     backend->gamma_shed(gamma_wing.d_view(), gamma_wing_prev.d_view(), gamma_wake.d_view(), i);
    //     // }

    //     // // Structure
    //     // {
    //     //     // TODO: map panel_forces_ip0, ip1 to structure domain
    //     //     auto F_ip0 = F.d_view().layout.slice(F.d_view().ptr, all, i);
    //     //     auto F_ip1 = F.d_view().layout.slice(F.d_view().ptr, all, i+1);
    //     //     // integrator.step(M.d_view(), C.d_view(), K.d_view(), F_ip0, F_ip1, dt, i);

    //     //     // parallel for
    //     //     for (i64 m = 0; m < nb_meshes; m++) {
    //     //         const auto local_transform = dual_to_float(kinematics[m].displacement(t));
    //     //         body_frames[m] = local_transform;
    //     //         local_transform.store(transforms.h_view().ptr + m*transforms.h_view().layout.stride(2), transforms.h_view().layout.stride(1));
    //     //     }
    //     //     transforms.to_device();
    //     //     backend->displace_wing(transforms.d_view(), mesh.verts_wing.d_view(), verts_wing_pos.d_view());

    //     // }
    // }
}

int main() {
    const i64 ni = 20;
    const i64 nj = 5;
    // vlm::Executor::instance(1);
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_" + std::to_string(ni) + "x" + std::to_string(nj) + ".x"};
    // const std::vector<std::string> meshes = {"../../../../mesh/rectangular_4x4.x"};
    
    const std::vector<std::string> backends = {"cpu"};

    auto simulations = tiny::make_combination(meshes, backends);

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
        linalg::float3{0.0f, 0.0f, 0.0f}, // take into account quarter chord panel offset
        linalg::float3{0.0f, 1.0f, 0.0f},
        to_radians(initial_angle)
    );
    
    // Sudden acceleration
    // const f32 alpha = to_radians(5.0f);
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({
    //         -u_inf*std::cos(alpha)*t,
    //         0.0f,
    //         -u_inf*std::sin(alpha)*t
    //     });
    // });

    for (const auto& [mesh_name, backend_name] : simulations) {
        // UVLM_2DOF simulation{backend_name, {mesh_name}};
        // simulation.run({kinematics}, {initial_pose}, t_final);
    }
    return 0;
}
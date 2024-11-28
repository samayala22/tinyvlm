#include "vlm.hpp"

#include "vlm_backend.hpp"
#include "tinytimer.hpp"
#include "tinyconfig.hpp"
#include "tinypbar.hpp"

#include "vlm_data.hpp"
#include "vlm_mesh.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"

#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace vlm;

Simulation::Simulation(const std::string& backend_name, const std::vector<std::string>& meshes) : backend(create_backend(backend_name)) {
    // Read the sizes of all the meshes
    for (const auto& m_name : meshes) {
        const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
        auto [nc,ns] = mesh_io.get_dims(m_name);
        assembly_wings.push_back({ns, nc});
    }

    MultiDim<3> verts_wing_3D;
    for (const auto& [ns, nc] : assembly_wings) {
        verts_wing_3D.push_back({ns+1, nc+1, 4});
    }

    verts_wing_init.init(verts_wing_3D);
    verts_wing_init_h.init(verts_wing_3D);
    verts_wing.init(verts_wing_3D);
    verts_wing_h.init(verts_wing_3D);

    // Perform the actual read of the mesh files
    for (i64 i = 0; i < meshes.size(); i++) {
        const MeshIO mesh_io{"plot3d"};
        mesh_io.read(meshes[i], verts_wing_init_h.views()[i]);
    }

    for (const auto& [init_h, init_d] : zip(verts_wing_init_h.views(), verts_wing_init.views())) {
        init_h.slice(All, All, 3).fill(1.f);
        init_h.to(init_d);
    }
};

inline i64 total_panels(const MultiDim<2>& assembly_wing) {
    i64 total = 0;
    for (const auto& wing : assembly_wing) {
        total += wing[0] * wing[1];
    }
    return total;
}

VLM::VLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    solver = backend->create_lu_solver();
    alloc_buffers();
};

void VLM::alloc_buffers() {
    const i64 n = total_panels(assembly_wings);
    condition0.resize(assembly_wings.size()*assembly_wings.size());
    MultiDim<3> panels_3D;
    MultiDim<2> panels_2D;
    MultiDim<2> wake_panels_2D;
    MultiDim<3> verts_wake_3D;
    MultiDim<2> transforms_2D;

    for (const auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
        panels_2D.push_back({ns, nc});
        wake_panels_2D.push_back({ns, 1});
        verts_wake_3D.push_back({ns+1, 2, 4}); // single row of wake panels
        transforms_2D.push_back({4, 4});
    }

    // Allocate
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    areas_d.init(panels_2D);
    verts_wake.init(verts_wake_3D);
    lhs.init({n, n});
    rhs.init({n});
    gamma_wing.init(panels_2D);
    gamma_wing_prev.init(panels_2D);
    gamma_wing_delta.init(panels_2D);
    gamma_wake.init(wake_panels_2D);
    local_velocities.init(panels_3D);
    wake_transform.init({4,4});
    transforms.init(transforms_2D);
    solver->init(lhs.view());

    for (const auto& wake : verts_wake.views()) wake.slice(All, All, 3).fill(1.f);
}

void print(const TensorView2D<Location::Device>& tensor) {
    for (i64 i = 0; i < tensor.shape(0); i++) {
        for (i64 j = 0; j < tensor.shape(1); j++) {
            std::printf("%6.6f ", tensor(i, j));
        }
        std::printf("\n");
    }
}

void print(const char* name, const MultiTensorView3D<Location::Device>& tensor) {
    std::printf("%s\n", name);
    for (i64 m = 0; m < tensor.size(); m++) {
        std::printf("Surface %lld\n", m);
        for (i64 j = 0; j < tensor[m].shape(1); j++) {
            for (i64 i = 0; i < tensor[m].shape(0); i++) {
                std::printf("x: %6.6f y: %6.6f z: %6.6f\n", tensor[m](i, j, 0), tensor[m](i, j, 1), tensor[m](i, j, 2));
            }
        }
    }
}

template<Location L>
void print(const char* name, const TensorView2D<L>& tensor) {
    std::printf("%s\n", name);
    for (i64 j = 0; j < tensor.shape(1); j++) {
        for (i64 i = 0; i < tensor.shape(0); i++) {
            std::printf("i: %6lld j: %6lld %6.6f\n", i, j, tensor(i, j));
        }
    }
}

template<Location L>
void print(const char* name, const TensorView3D<L>& tensor) {
    std::printf("%s\n", name);
    for (i64 j = 0; j < tensor.shape(1); j++) {
        for (i64 i = 0; i < tensor.shape(0); i++) {
            std::printf("x: %6.6f y: %6.6f z: %6.6f\n", tensor(i, j, 0), tensor(i, j, 1), tensor(i, j, 2));
        }
    }
}

void print(const char* name, const MultiTensorView2D<Location::Device>& tensor) {
    std::printf("%s\n", name);
    for (i64 m = 0; m < tensor.size(); m++) {
        const auto& tensor_m = tensor[m];
        for (i64 j = 0; j < tensor_m.shape(1); j++) {
            for (i64 i = 0; i < tensor_m.shape(0); i++) {
                std::printf("%6.6f \n", tensor_m(i, j));
            }
        }
    }
}

AeroCoefficients VLM::run(const FlowData& flow) {
    // Reset buffer state
    lhs.view().fill(0.f);
    rhs.view().fill(0.f);
    for (const auto& [wing_init, wing] : zip(verts_wing_init.views(), verts_wing.views())) wing_init.to(wing);
    
    // Move wing to create 100 chord long wake panels 
    auto init_pos = translation_matrix<f32>({
        -100.0f * flow.u_inf*std::cos(flow.alpha),
        0.0f,
        -100.0f * flow.u_inf*std::sin(flow.alpha)
    });
    init_pos.store(wake_transform.view().ptr(), wake_transform.view().stride(1));
    for (const auto& transform_d : transforms.views()) wake_transform.view().to(transform_d);

    backend->wake_shed(verts_wing.views(), verts_wake.views(), 0);
    backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
    backend->wake_shed(verts_wing.views(), verts_wake.views(), 1);
    
    backend->mesh_metrics(flow.alpha, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0,1);
    for (const auto& vel : local_velocities.views()) {
        vel.slice(All, All, 0).fill(flow.freestream.x);
        vel.slice(All, All, 1).fill(flow.freestream.y);
        vel.slice(All, All, 2).fill(flow.freestream.z);
    }
    backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), local_velocities.views());

    solver->factorize(lhs.view());
    solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
    
    i64 begin = 0;
    for (i64 m = 0; m < assembly_wings.size(); m++) {
        const auto& gamma_wing_i = gamma_wing.views()[m];
        const auto& gamma_wing_delta_i = gamma_wing_delta.views()[m];
        const auto& gamma_wing_prev_i = gamma_wing_prev.views()[m];
        const auto& gamma_wake_i = gamma_wake.views()[m];
        i64 end = begin + gamma_wing_i.size();
        rhs.view().slice(Range{begin, end}).to(gamma_wing_i.reshape(gamma_wing_i.size()));
        gamma_wing_i.to(gamma_wing_prev_i); // save prev iteration
        gamma_wing_i.to(gamma_wing_delta_i);
        gamma_wing_i.slice(All, -1).to(gamma_wake_i.slice(All, -1)); // shed to wake
        backend->blas->axpy(-1.0f, gamma_wing_i.slice(All, Range{0, -2}), gamma_wing_delta_i.slice(All, Range{1, -1}));
        begin = end;
    }

    return AeroCoefficients{
        backend->coeff_steady_cl_multi(verts_wing.views(), gamma_wing_delta.views(), flow, areas_d.views()),
        backend->coeff_steady_cd_multi(verts_wake.views(), gamma_wake.views(), flow, areas_d.views()),
        {0.0f, 0.0f, 0.0f} // todo implement cm
    };
}

NLVLM::NLVLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    solver = backend->create_lu_solver();
    alloc_buffers();
};

void NLVLM::alloc_buffers() {
    const i64 n = total_panels(assembly_wings);
    condition0.resize(assembly_wings.size()*assembly_wings.size());
    MultiDim<3> panels_3D;
    MultiDim<2> panels_2D;
    MultiDim<2> wake_panels_2D;
    MultiDim<3> verts_wake_3D;
    MultiDim<1> spanwise_1D;
    MultiDim<2> transforms_2D;

    for (const auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
        panels_2D.push_back({ns, nc});
        wake_panels_2D.push_back({ns, 1});
        verts_wake_3D.push_back({ns+1, 2, 4}); // single row of wake panels
        spanwise_1D.push_back({ns});
        transforms_2D.push_back({4, 4});
    }

    // Allocate
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    areas_d.init(panels_2D);
    verts_wake.init(verts_wake_3D);
    lhs.init({n, n});
    rhs.init({n});
    gamma_wing.init(panels_2D);
    gamma_wing_prev.init(panels_2D);
    gamma_wing_delta.init(panels_2D);
    gamma_wake.init(wake_panels_2D);
    local_velocities.init(panels_3D);
    strip_alphas.init(spanwise_1D);
    wake_transform.init({4,4});
    transforms.init(transforms_2D);
    
    solver->init(lhs.view());
    
    for (const auto& wake : verts_wake.views()) wake.slice(All, All, 3).fill(1.f);
}

void strip_alpha_to_vel(const FlowData& flow, MultiTensorView3D<Location::Device>& local_velocities, const MultiTensorView1D<Location::Host>& strip_alphas) {
    for (const auto& [strip, vel] : zip(strip_alphas, local_velocities)) {
        // Note: this is not the most efficient way
        // we can set the first row and then copy each contiguous spanwise slice instead
        for (i64 i = 0; i < strip.shape(0); i++) {
            auto fs = compute_freestream(flow.u_inf, strip(i), 0.0f);
            vel.slice(i, All, 0).fill(fs.x);
            vel.slice(i, All, 1).fill(fs.y);
            vel.slice(i, All, 2).fill(fs.z);
        }
    }
}

AeroCoefficients NLVLM::run(const FlowData& flow, const Database& db) {
    f64 err = 1.0f; // l1 error
    auto init_pos = translation_matrix<f32>({
        -100.0f * flow.u_inf*std::cos(flow.alpha),
        0.0f,
        -100.0f * flow.u_inf*std::sin(flow.alpha)
    });

    lhs.view().fill(0.f);
    init_pos.store(wake_transform.view().ptr(), wake_transform.view().stride(1));
    for (const auto& transform_d : transforms.views()) wake_transform.view().to(transform_d);
    for (const auto& strip : strip_alphas.views()) strip.fill(flow.alpha);
    for (const auto& [wing_init, wing] : zip(verts_wing_init.views(), verts_wing.views())) wing_init.to(wing);
    
    backend->wake_shed(verts_wing.views(), verts_wake.views(), 0);
    backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
    backend->wake_shed(verts_wing.views(), verts_wake.views(), 1);
    backend->mesh_metrics(flow.alpha, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0,1);
    solver->factorize(lhs.view());

    for (i64 iter = 0; iter < max_iter && err > tol; iter++) {
        err = 0.0; // reset l1 error
        const auto& strip_alphas_v = strip_alphas.views();
        strip_alpha_to_vel(flow, local_velocities.views(), strip_alphas.views()); // Compute local panel velocities based on strip alphas
        rhs.view().fill(0.f);
        backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), local_velocities.views());
        solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
        
        i64 begin = 0;
        for (i64 m = 0; m < assembly_wings.size(); m++) {
            const auto& gamma_wing_i = gamma_wing.views()[m];
            const auto& gamma_wing_delta_i = gamma_wing_delta.views()[m];
            const auto& gamma_wing_prev_i = gamma_wing_prev.views()[m];
            const auto& gamma_wake_i = gamma_wake.views()[m];
            i64 end = begin + gamma_wing_i.size();
            rhs.view().slice(Range{begin, end}).to(gamma_wing_i.reshape(gamma_wing_i.size()));
            gamma_wing_i.to(gamma_wing_prev_i); // unused
            gamma_wing_i.to(gamma_wing_delta_i);
            gamma_wing_i.slice(All, -1).to(gamma_wake_i.slice(All, -1)); // unused
            backend->blas->axpy(-1.0f, gamma_wing_i.slice(All, Range{0, -2}), gamma_wing_delta_i.slice(All, Range{1, -1}));
            begin = end;
        }
 
        // Parallel Reduce
        // loop over the chordwise strips and apply Van Dam algorithm
        for (i64 m = 0; m < strip_alphas_v.size(); m++) {
            const auto& strip_alpha_m = strip_alphas_v[m];
            const auto& verts_wing_m = verts_wing.views()[m];
            const auto& gamma_wing_delta_m = gamma_wing_delta.views()[m];
            const auto& areas_m = areas_d.views()[m];
            for (i64 i = 0; i < strip_alphas_v[m].shape(0); i++) {
                auto verts_wing_i = verts_wing_m.slice(Range{i, i+2}, All, All);
                auto gamma_wing_delta_i = gamma_wing_delta_m.slice(Range{i, i+1}, All);
                auto areas_i = areas_m.slice(Range{i, i+1}, All);
                
                const FlowData strip_flow = {strip_alpha_m(i), flow.beta, flow.u_inf, flow.rho};
                const f32 strip_area = backend->sum(areas_i);
                const f32 strip_cl = backend->coeff_steady_cl_single(verts_wing_i, gamma_wing_delta_i, strip_flow, strip_area);
                const f32 effective_aoa = strip_cl / (2.f*PI_f) - strip_flow.alpha + flow.alpha;

                // TODO: interpolated value should be computed at the y mid point of the strip
                const f32 correction = (db.interpolate_CL(effective_aoa, 0.f) - strip_cl) / (2.f*PI_f);
                // std::printf(">>> Strip: %llu | CL: %.3f | Interpolated: %.3f | Correction: %.3e\n", i, strip_cl, db.interpolate_CL(effective_aoa, 0.f), correction);
                strip_alpha_m(i) += correction;
                err += (f64)std::abs(correction);
            }
        }
        err /= (f64)total_panels(assembly_wings);; // normalize l1 error
        //std::printf(">>> Iter: %d | Error: %.3e \n", iter, err);
    }

    return AeroCoefficients{
        backend->coeff_steady_cl_multi(verts_wing.views(), gamma_wing_delta.views(), flow, areas_d.views()),
        backend->coeff_steady_cd_multi(verts_wake.views(), gamma_wake.views(), flow, areas_d.views()),
        {0.0f, 0.0f, 0.0f} // todo implement
    };
}

UVLM::UVLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    solver = backend->create_lu_solver();
    alloc_buffers();
}

void UVLM::alloc_buffers() {
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
    transforms_h.init(transforms_2D);
    transforms.init(transforms_2D);
    solver->init(lhs.view());

    condition0.resize(assembly_wings.size()*assembly_wings.size());
}

void UVLM::run(const Assembly& assembly, f32 t_final) {
    const tiny::ScopedTimer timer("UVLM");
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
    lhs.view().fill(0.f);

    // 1.  Compute the fixed time step
    const auto& verts_first_wing = verts_wing_init_h.views()[0];
    const f32 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    const f32 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 1);
    const f32 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 2);
    const f32 last_panel_chord = std::sqrt(dx*dx + dy*dy + dz*dz);
    const f32 dt = last_panel_chord / linalg::length(assembly.kinematics()->linear_velocity(0.0f, {0.f, 0.f, 0.f}));
    const i64 t_steps = static_cast<i64>(t_final / dt);
    
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

    // 3. Precompute constant values for the transient simulation
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0 , 0);
    solver->factorize(lhs.view());

    std::ofstream cl_data("cl_data.txt");
    cl_data << "0.5" << "\n"; // TODO: this should be according to inputs

    // 4. Transient simulation loop
    // for (i32 i = 0; i < vec_t.size()-1; i++) {
    for (const i32 i : tiny::pbar(0, (i32)t_steps-1)) {
        const f32 t = (f32)i * dt;
        t_h.view()(i) = t;
        
        const linalg::float3 freestream = -assembly.kinematics()->linear_velocity(t, {0.f, 0.f, 0.f});

        backend->wake_shed(verts_wing.views(), verts_wake.views(), i);

        // parallel for
        for (i64 m = 0; m < assembly_wings.size(); m++) {
            const auto transform_node = assembly.surface_kinematics()[m];
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

        rhs.view().fill(0.f);
        backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.views());
        backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.views(), colloc_d.views(), normals_d.views(), verts_wake.views(), i);
        solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
        
        i64 begin = 0;
        for (i64 m = 0; m < assembly_wings.size(); m++) {
            const auto& gamma_wing_i = gamma_wing.views()[m];
            const auto& gamma_wing_delta_i = gamma_wing_delta.views()[m];
            const auto& gamma_wake_i = gamma_wake.views()[m];
            i64 end = begin + gamma_wing_i.size();
            rhs.view().slice(Range{begin, end}).to(gamma_wing_i.reshape(gamma_wing_i.size()));
            gamma_wing_i.to(gamma_wing_delta_i);
            gamma_wing_i.slice(All, -1).to(gamma_wake_i.slice(All, -1-i)); // shed to wake
            backend->blas->axpy(-1.0f, gamma_wing_i.slice(All, Range{0, -2}), gamma_wing_delta_i.slice(All, Range{1, -1}));
            begin = end;
        }
        
        // skip cl computation for the first iteration
        if (i > 0) {
            const f32 cl_unsteady = backend->coeff_unsteady_cl_multi(verts_wing.views(), gamma_wing_delta.views(), gamma_wing.views(), gamma_wing_prev.views(), velocities.views(), areas_d.views(), normals_d.views(), freestream, dt);
            // if (i == vec_t.size()-2) std::printf("t: %f, CL: %f\n", t, cl_unsteady);
            std::printf("t: %f, CL: %f\n", t, cl_unsteady);
            for (const auto& [wing, wing_h] : zip(verts_wing.views(), verts_wing_h.views())) {
                wing.to(wing_h);
            }
            cl_data << t << " " << verts_wing_h.views()[0](0, 0, 2) << " " << cl_unsteady << " " << 0.f << "\n";
        }

        {
            // parallel for
            i64 begin = 0;
            for (i64 m = 0; m < assembly_wings.size(); m++) {
                const auto& gamma_wing_i = gamma_wing.views()[m];
                const auto& gamma_wing_prev_i = gamma_wing_prev.views()[m];

                i64 end = begin + gamma_wing_i.size();
                gamma_wing_i.to(gamma_wing_prev_i); // shed gamma

                auto transform = assembly.surface_kinematics()[m]->transform(t+dt);
                transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
                transforms_h.views()[m].to(transforms.views()[m]);
                begin = end;
            }
        }

        backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_pos.views());
        backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    }
}
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
    i64 off_wing_p = 0;
    i64 off_wing_v = 0;
    for (const auto& m_name : meshes) {
        const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
        auto [nc,ns] = mesh_io.get_dims(m_name);
        assembly_wings.emplace_back(ns, nc);
        wing_panels.emplace_back(SurfaceDims{nc, ns, off_wing_p});
        wing_vertices.emplace_back(SurfaceDims{nc+1, ns+1, off_wing_v});
        off_wing_p += wing_panels.back().size();
        off_wing_v += wing_vertices.back().size();
    }

    mesh.alloc_wing(wing_panels, wing_vertices);
    nb_meshes = meshes.size();
    wing_positions.resize(meshes.size());

    // Perform the actual read of the mesh files
    for (i64 i = 0; i < meshes.size(); i++) {
        const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
        auto& current_view = mesh.verts_wing_init.h_view();
        View<f32, SingleSurface> vertices = current_view.layout.subview(current_view.ptr, i, 0, current_view.layout.nc(i), 0, current_view.layout.ns(i));
        
        mesh_io.read(meshes[i], vertices);
    }
};

VLM::VLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes), 
    lhs(*backend->memory), rhs(*backend->memory), gamma_wing(*backend->memory), gamma_wake(*backend->memory), gamma_wing_prev(*backend->memory), gamma_wing_delta(*backend->memory), local_velocities(*backend->memory), transforms(*backend->memory) {
    
    solver = backend->create_lu_solver();

    const i64 nw = 1;
    i64 off_wake_p = 0;
    i64 off_wake_v = 0;
    for (i32 i = 0; i < wing_panels.size(); i++) {
        wake_panels.emplace_back(SurfaceDims{nw, wing_panels[i].ns, off_wake_p});
        wake_vertices.emplace_back(SurfaceDims{nw+1, wing_vertices[i].ns, off_wake_v});
        off_wake_p += wake_panels.back().size();
        off_wake_v += wake_vertices.back().size();
    }

    mesh.verts_wing_init.to_device();
    mesh.alloc_wake(wake_vertices);  
    alloc_buffers();
};

void VLM::alloc_buffers() {
    const i64 n = wing_panels.back().offset + wing_panels.back().size();
    condition0.resize(wing_panels.size());
    MultiDim<3> panels_3D;
    for (auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
    }
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    lhs.init({n, n});
    rhs.init({n});
    gamma_wing.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_prev.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_delta.alloc(MultiSurface{wing_panels, 1});
    gamma_wake.alloc(MultiSurface{wake_panels, 1});
    local_velocities.alloc(MultiSurface{wing_panels, 3});
    wake_transform.init({4,4});
    transforms.init({4,4,nb_meshes});
    solver->init(lhs.view());
}

AeroCoefficients VLM::run(const FlowData& flow) {
    // Reset buffer state
    lhs.view().fill(0.f);
    rhs.view().fill(0.f);
    backend->memory->copy(Location::Device, mesh.verts_wing.d_view().ptr, 1, Location::Device, mesh.verts_wing_init.d_view().ptr, 1, sizeof(f32), mesh.verts_wing_init.d_view().size());
    // Move wing to create 100 chord long wake panels 
    auto init_pos = translation_matrix<f32>({
        -100.0f * flow.u_inf*std::cos(flow.alpha),
        0.0f,
        -100.0f * flow.u_inf*std::sin(flow.alpha)
    });
    init_pos.store(wake_transform.view().ptr(), wake_transform.view().stride(1));
    for (i32 i = 0; i < transforms.view().shape(2); i++) {
        auto transform = transforms.view().slice(All, All, i);
        wake_transform.view().to(transform);
    }

    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 0);
    backend->displace_wing(transforms.view(), mesh.verts_wing.d_view(), mesh.verts_wing_init.d_view());
    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 1);
    
    backend->mesh_metrics(flow.alpha, mesh.verts_wing.d_view(), colloc_d.views(), normals_d.views(), mesh.area.d_view());
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), condition0,1);    
    backend->memory->fill(Location::Device, local_velocities.d_view().ptr + 0 * local_velocities.d_view().layout.stride(), 1, flow.freestream.x, local_velocities.d_view().layout.stride());
    backend->memory->fill(Location::Device, local_velocities.d_view().ptr + 1 * local_velocities.d_view().layout.stride(), 1, flow.freestream.y, local_velocities.d_view().layout.stride());
    backend->memory->fill(Location::Device, local_velocities.d_view().ptr + 2 * local_velocities.d_view().layout.stride(), 1, flow.freestream.z, local_velocities.d_view().layout.stride());
    backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), local_velocities.d_view());

    solver->factorize(lhs.view());
    solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
    backend->memory->copy(Location::Device, gamma_wing.d_view().ptr, 1, Location::Device, rhs.view().ptr(), 1, sizeof(f32), rhs.size());

    backend->gamma_shed(gamma_wing.d_view(), gamma_wing_prev.d_view(), gamma_wake.d_view(), 0);
    backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());

    return AeroCoefficients{
        backend->coeff_steady_cl_multi(mesh.verts_wing.d_view(), gamma_wing_delta.d_view(), flow, mesh.area.d_view()),
        backend->coeff_steady_cd_multi(mesh.verts_wake.d_view(), gamma_wake.d_view(), flow, mesh.area.d_view()),
        {0.0f, 0.0f, 0.0f} // todo implement cm
    };
}

NLVLM::NLVLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes), 
    lhs(*backend->memory), rhs(*backend->memory), gamma_wing(*backend->memory), gamma_wake(*backend->memory), gamma_wing_prev(*backend->memory), gamma_wing_delta(*backend->memory), local_velocities(*backend->memory), strip_alphas(*backend->memory), transforms(*backend->memory) {
    
    solver = backend->create_lu_solver();

    const i64 nw = 1;
    i64 off_wake_p = 0;
    i64 off_wake_v = 0;
    for (i32 i = 0; i < wing_panels.size(); i++) {
        wake_panels.emplace_back(SurfaceDims{nw, wing_panels[i].ns, off_wake_p});
        wake_vertices.emplace_back(SurfaceDims{nw+1, wing_vertices[i].ns, off_wake_v});
        off_wake_p += wake_panels.back().size();
        off_wake_v += wake_vertices.back().size();
    }

    mesh.verts_wing_init.to_device();
    mesh.alloc_wake(wake_vertices);
    alloc_buffers();
};

void NLVLM::alloc_buffers() {
    const i64 n = wing_panels.back().offset + wing_panels.back().size();
    condition0.resize(wing_panels.size());
    MultiDim<3> panels_3D;
    for (auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
    }
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    lhs.init({n, n});
    rhs.init({n});
    gamma_wing.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_prev.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_delta.alloc(MultiSurface{wing_panels, 1});
    gamma_wake.alloc(MultiSurface{wake_panels, 1});
    local_velocities.alloc(MultiSurface{wing_panels, 3});
    std::vector<SurfaceDims> strip_alpha_layout = wing_panels;
    for (auto& dims : strip_alpha_layout) dims.nc = 1;
    strip_alphas.alloc(MultiSurface{strip_alpha_layout, 1});
    wake_transform.init({4,4});
    transforms.init({4,4,nb_meshes});
    solver->init(lhs.view());
}

void strip_alpha_to_vel(const FlowData& flow, View<f32, MultiSurface>& local_velocities, const View<f32, MultiSurface>& strip_alphas) {
    for (i64 m = 0; m < local_velocities.layout.surfaces().size(); m++) {
        assert(strip_alphas.layout.ns(m) == local_velocities.layout.ns(m));
        f32* strip_alphas_m = strip_alphas.ptr + strip_alphas.layout.offset(m);
        f32* local_velocities_m = local_velocities.ptr + local_velocities.layout.offset(m);
        // todo: this can be done with fill (eliminating the need for mem transfer)
        for (i64 j = 0; j < strip_alphas.layout.ns(m); j++) {
            auto fs = compute_freestream(flow.u_inf, strip_alphas_m[j], 0.0f);
            local_velocities_m[0*local_velocities.layout.stride() + j] = fs.x;
            local_velocities_m[1*local_velocities.layout.stride() + j] = fs.y;
            local_velocities_m[2*local_velocities.layout.stride() + j] = fs.z;
        }
        for (i64 i = 1; i < local_velocities.layout.nc(m); i++) {
            std::memcpy(local_velocities_m + 0 * local_velocities.layout.stride() + i * local_velocities.layout.ns(m), local_velocities_m, local_velocities.layout.ns(m) * sizeof(*local_velocities_m));
            std::memcpy(local_velocities_m + 1 * local_velocities.layout.stride() + i * local_velocities.layout.ns(m), local_velocities_m + 1 * local_velocities.layout.stride(), local_velocities.layout.ns(m) * sizeof(*local_velocities_m));
            std::memcpy(local_velocities_m + 2 * local_velocities.layout.stride() + i * local_velocities.layout.ns(m), local_velocities_m + 2 * local_velocities.layout.stride(), local_velocities.layout.ns(m) * sizeof(*local_velocities_m));
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
    init_pos.store(wake_transform.view().ptr(), wake_transform.view().stride(1));
    for (i32 i = 0; i < transforms.view().shape(2); i++) {
        auto transform = transforms.view().slice(All, All, i);
        wake_transform.view().to(transform);
    }
    lhs.view().fill(0.f);
    backend->memory->fill(Location::Host, strip_alphas.h_view().ptr, 1, flow.alpha, strip_alphas.h_view().size());
    backend->memory->copy(Location::Device, mesh.verts_wing.d_view().ptr, 1, Location::Device, mesh.verts_wing_init.d_view().ptr, 1, sizeof(f32), mesh.verts_wing_init.d_view().size());
    
    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 0);
    backend->displace_wing(transforms.view(), mesh.verts_wing.d_view(), mesh.verts_wing_init.d_view());
    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 1);
    backend->mesh_metrics(flow.alpha, mesh.verts_wing.d_view(), colloc_d.views(), normals_d.views(), mesh.area.d_view());
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), condition0,1);
    solver->factorize(lhs.view());

    for (i64 iter = 0; iter < max_iter && err > tol; iter++) {
        err = 0.0; // reset l1 error
        strip_alpha_to_vel(flow, local_velocities.h_view(), strip_alphas.h_view()); // Compute local panel velocities based on strip alphas
        local_velocities.to_device();
        rhs.view().fill(0.f);
        backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), local_velocities.d_view());
        solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
        backend->memory->copy(Location::Device, gamma_wing.d_view().ptr, 1, Location::Device, rhs.view().ptr(), 1, sizeof(f32), rhs.view().size());

        backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());
 
        // Parallel Reduce
        // loop over the chordwise strips and apply Van Dam algorithm
        for (i64 m = 0; m < strip_alphas.h_view().layout.surfaces().size(); m++) {
            for (i64 j = 0; j < strip_alphas.h_view().layout.ns(m); j++) {
                f32* strip_alphas_m = strip_alphas.h_view().ptr + strip_alphas.h_view().layout.offset(m);
                auto area_strip = mesh.area.d_view().layout.subview(mesh.area.d_view().ptr, m, 0, mesh.area.d_view().layout.nc(m), j, 1);
                auto verts_wing_strip = mesh.verts_wing.d_view().layout.subview(mesh.verts_wing.d_view().ptr, m, 0, mesh.verts_wing.d_view().layout.nc(m), j, 2);
                auto gamma_delta_strip = gamma_wing_delta.d_view().layout.subview(gamma_wing_delta.d_view().ptr, m, 0, gamma_wing_delta.d_view().layout.nc(m), j, 1);
                
                const f32 strip_area = backend->mesh_area(area_strip);
                const FlowData strip_flow = {strip_alphas_m[j], flow.beta, flow.u_inf, flow.rho};
                const f32 strip_cl = backend->coeff_steady_cl_single(verts_wing_strip, gamma_delta_strip, strip_flow, strip_area);
                const f32 effective_aoa = strip_cl / (2.f*PI_f) - strip_flow.alpha + flow.alpha;

                // TODO: interpolated value should be computed at the y mid point of the strip
                const f32 correction = (db.interpolate_CL(effective_aoa, 0.f) - strip_cl) / (2.f*PI_f);
                // std::printf(">>> Strip: %llu | CL: %.3f | Interpolated: %.3f | Correction: %.3e\n", j, strip_cl, db.interpolate_CL(effective_aoa, 0.f), correction);
                strip_alphas_m[j] += correction;
                err += (f64)std::abs(correction);
            }
        }
        err /= strip_alphas.h_view().size(); // normalize l1 error
        //std::printf(">>> Iter: %d | Error: %.3e \n", iter, err);
    }
    backend->gamma_shed(gamma_wing.d_view(), gamma_wing_prev.d_view(), gamma_wake.d_view(), 0);

    return AeroCoefficients{
        backend->coeff_steady_cl_multi(mesh.verts_wing.d_view(), gamma_wing_delta.d_view(), flow, mesh.area.d_view()),
        backend->coeff_steady_cd_multi(mesh.verts_wake.d_view(), gamma_wake.d_view(), flow, mesh.area.d_view()),
        {0.0f, 0.0f, 0.0f} // todo implement
    };
}

UVLM::UVLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    solver = backend->create_lu_solver();
    alloc_buffers();
}

void UVLM::alloc_buffers() {
    const i64 n = wing_panels.back().offset + wing_panels.back().size();

    // Mesh
    MultiDim<3> panels_3D;
    for (auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
    }
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    verts_wing_pos.alloc(MultiSurface{wing_vertices, 4});
    colloc_pos.alloc(MultiSurface{wing_panels, 3});

    // Data
    lhs.init({n, n});
    rhs.init({n});
    gamma_wing.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_prev.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_delta.alloc(MultiSurface{wing_panels, 1});
    velocities.alloc(MultiSurface{wing_panels, 3});
    transforms_h.init({4,4,nb_meshes});
    transforms.init({4,4,nb_meshes});
    solver->init(lhs.view());

    local_dt.resize(nb_meshes);
    condition0.resize(nb_meshes);
}

void UVLM::run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::alias::float4x4>& initial_pose, f32 t_final) {
    const tiny::ScopedTimer timer("UVLM");
    mesh.verts_wing_init.to_device(); // raw mesh vertices from file
    for (i64 m = 0; m < kinematics.size(); m++) {
        initial_pose[m].store(transforms_h.view().ptr() + transforms_h.view().offset({0, 0, m}), transforms_h.view().stride(1));
    }
    transforms_h.view().to(transforms.view());
    backend->displace_wing(transforms.view(), verts_wing_pos.d_view(), mesh.verts_wing_init.d_view());
    
    for (i32 body = 0; body < nb_meshes; body++) {
        backend->memory->copy(Location::Host, colloc_pos.h_view().ptr + colloc_pos.h_view().layout.offset(body), 1, Location::Device, colloc_d.views()[body].ptr(), 1, sizeof(f32), colloc_d.views()[body].size());
    }
    backend->memory->copy(Location::Device, mesh.verts_wing.d_view().ptr, 1, Location::Device, verts_wing_pos.d_view().ptr, 1, sizeof(f32), verts_wing_pos.d_view().size());
    lhs.view().fill(0.f);
    backend->mesh_metrics(0.0f, mesh.verts_wing.d_view(), colloc_d.views(), normals_d.views(), mesh.area.d_view());

    // 1.  Compute the dynamic time steps for the simulation
    verts_wing_pos.to_host();
    vec_t.clear();
    vec_t.push_back(0.0f);
    for (f32 t = 0.0f; t < t_final;) {
        // parallel for
        for (i64 m = 0; m < nb_meshes; m++) {
            local_dt[m] = std::numeric_limits<f32>::max();
            const auto local_transform = kinematics[m].displacement(t);
            const f32* local_verts = verts_wing_pos.h_view().ptr + verts_wing_pos.h_view().layout.offset(m);
            const i64 pre_trailing_begin = (verts_wing_pos.h_view().layout.nc(m)-2)*verts_wing_pos.h_view().layout.ns(m);
            const i64 trailing_begin = (verts_wing_pos.h_view().layout.nc(m)-1)*verts_wing_pos.h_view().layout.ns(m);
            // parallel for
            for (i64 j = 0; j < verts_wing_pos.h_view().layout.ns(m); j++) {
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
        for (i64 m = 0; m < kinematics.size(); m++) {
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
        
        const i64 nw = vec_t.size()-1;
        i64 off_wake_p = 0;
        i64 off_wake_v = 0;
        for (i32 i = 0; i < wing_panels.size(); i++) {
            wake_panels.emplace_back(SurfaceDims{nw, wing_panels[i].ns, off_wake_p});
            wake_vertices.emplace_back(SurfaceDims{nw+1, wing_vertices[i].ns, off_wake_v});
            off_wake_p += wake_panels.back().size();
            off_wake_v += wake_vertices.back().size();
        }
        gamma_wake.alloc(MultiSurface{wake_panels, 1});
        mesh.alloc_wake(wake_vertices);
    }

    // 3. Precompute constant values for the transient simulation
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), condition0 , 0);
    solver->factorize(lhs.view());

    std::ofstream cl_data("cl_data.txt");
    cl_data << "0.5" << "\n";

    // 4. Transient simulation loop
    // for (i32 i = 0; i < vec_t.size()-1; i++) {
    for (const i32 i : tiny::pbar(0, (i32)vec_t.size()-1)) {
        const f32 t = vec_t[i];
        const f32 dt = vec_t[i+1] - t;

        const linalg::alias::float3 freestream = -kinematics[0].velocity(kinematics[0].displacement(t,1), {0.f, 0.f, 0.f, 1.0f});

        backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), i);

        // parallel for
        for (i64 m = 0; m < nb_meshes; m++) {
            const auto local_transform = kinematics[m].displacement(t);
            const f32* local_colloc = colloc_pos.h_view().ptr + colloc_pos.h_view().layout.offset(m);
            f32* local_velocities = velocities.h_view().ptr + velocities.h_view().layout.offset(m);
            
            // parallel for
            for (i64 idx = 0; idx < colloc_pos.h_view().layout.surface(m).size(); idx++) {
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
        rhs.view().fill(0.f);
        backend->rhs_assemble_velocities(rhs.view(), normals_d.views(), velocities.d_view());
        backend->rhs_assemble_wake_influence(rhs.view(), gamma_wake.d_view(), colloc_d.views(), normals_d.views(), mesh.verts_wake.d_view(), i);
        solver->solve(lhs.view(), rhs.view().reshape(rhs.size(), 1));
        backend->memory->copy(Location::Device, gamma_wing.d_view().ptr, 1, Location::Device, rhs.view().ptr(), 1, sizeof(f32), rhs.view().size());

        backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());
        
        // skip cl computation for the first iteration
        if (i > 0) {
            const f32 cl_unsteady = backend->coeff_unsteady_cl_multi(mesh.verts_wing.d_view(), gamma_wing_delta.d_view(), gamma_wing.d_view(), gamma_wing_prev.d_view(), velocities.d_view(), mesh.area.d_view(), normals_d.views(), freestream, dt);
            // if (i == vec_t.size()-2) std::printf("t: %f, CL: %f\n", t, cl_unsteady);
            // std::printf("t: %f, CL: %f\n", t, cl_unsteady);
            mesh.verts_wing.to_host();
            cl_data << t << " " << *(mesh.verts_wing.h_view().ptr + 2*mesh.verts_wing.h_view().layout.stride()) << " " << cl_unsteady << " " << 0.f << "\n";
        }

        // backend->displace_wake_rollup(wake_rollup.d_view(), mesh.verts_wake.d_view(), mesh.verts_wing.d_view(), gamma_wing.d_view(), gamma_wake.d_view(), dt, i);
        backend->gamma_shed(gamma_wing.d_view(), gamma_wing_prev.d_view(), gamma_wake.d_view(), i);
        
        // parallel for
        for (i64 m = 0; m < nb_meshes; m++) {
            const auto local_transform = dual_to_float(kinematics[m].displacement(t+dt));
            local_transform.store(transforms_h.view().ptr() + transforms_h.view().offset({0, 0, m}), transforms_h.view().stride(1));
        }
        transforms_h.view().to(transforms.view());
        backend->displace_wing(transforms.view(), mesh.verts_wing.d_view(), verts_wing_pos.d_view());
        backend->mesh_metrics(0.0f, mesh.verts_wing.d_view(), colloc_d.views(), normals_d.views(), mesh.area.d_view());
    }
}
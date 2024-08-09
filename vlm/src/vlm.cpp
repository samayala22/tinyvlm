#include "vlm.hpp"

#include "vlm_backend.hpp"
#include "tinytimer.hpp"
#include "tinyconfig.hpp"

#include "vlm_data.hpp"
#include "vlm_mesh.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"

#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace vlm;

Simulation::Simulation(const std::string& backend_name, const std::vector<std::string>& meshes) : backend(create_backend(backend_name)), mesh(*backend->memory) {
    // Read the sizes of all the meshes
    u64 off_wing_p = 0;
    u64 off_wing_v = 0;
    for (const auto& m_name : meshes) {
        const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
        auto [nc,ns] = mesh_io.get_dims(m_name);
        wing_panels.emplace_back(SurfaceDims{nc, ns, off_wing_p});
        wing_vertices.emplace_back(SurfaceDims{nc+1, ns+1, off_wing_v});
        off_wing_p += wing_panels.back().size();
        off_wing_v += wing_vertices.back().size();
    }

    mesh.alloc_wing(wing_panels, wing_vertices);
    wing_positions.resize(meshes.size());

    // Perform the actual read of the mesh files
    for (u64 i = 0; i < meshes.size(); i++) {
        const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
        auto& current_view = mesh.verts_wing_init.h_view();
        View<f32, SingleSurface> vertices = current_view.layout.subview(current_view.ptr, i, 0, current_view.layout.nc(i), 0, current_view.layout.ns(i));
        
        mesh_io.read(meshes[i], vertices);
    }
};

VLM::VLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes), 
    lhs(*backend->memory), rhs(*backend->memory), gamma_wing(*backend->memory), gamma_wake(*backend->memory), gamma_wing_prev(*backend->memory), gamma_wing_delta(*backend->memory), local_velocities(*backend->memory) {
    
    const u64 nw = 1;
    u64 off_wake_p = 0;
    u64 off_wake_v = 0;
    for (u32 i = 0; i < wing_panels.size(); i++) {
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
    const u64 n = wing_panels.back().offset + wing_panels.back().size();
    condition0.resize(wing_panels.size());
    lhs.alloc(Matrix<MatrixLayout::ColMajor>{n, n, n});
    rhs.alloc(MultiSurface{wing_panels, 1});
    gamma_wing.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_prev.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_delta.alloc(MultiSurface{wing_panels, 1});
    gamma_wake.alloc(MultiSurface{wake_panels, 1});
    local_velocities.alloc(MultiSurface{wing_panels, 3});
}

AeroCoefficients VLM::run(const FlowData& flow) {
    // Reset buffer state
    backend->memory->fill_f32(MemoryLocation::Device, lhs.d_view().ptr, 0.f, lhs.d_view().size());
    backend->memory->fill_f32(MemoryLocation::Device, rhs.d_view().ptr, 0.f, rhs.d_view().size());
    backend->memory->copy(MemoryTransfer::DeviceToDevice, mesh.verts_wing.d_view().ptr, mesh.verts_wing_init.d_view().ptr, mesh.verts_wing.d_view().size_bytes());
    backend->lu_allocate(lhs.d_view());
    
    // global initial position
    auto init_pos = translation_matrix<f32>({
        -100.0f * flow.u_inf*std::cos(flow.alpha),
        0.0f,
        -100.0f * flow.u_inf*std::sin(flow.alpha)
    });
    std::fill(wing_positions.begin(), wing_positions.end(), init_pos);

    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 0);
    backend->displace_wing(wing_positions, mesh.verts_wing.d_view(), mesh.verts_wing_init.d_view());
    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 1);

    backend->mesh_metrics(flow.alpha, mesh.verts_wing.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.area.d_view());
    backend->lhs_assemble(lhs.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), condition0,1);
    backend->memory->fill_f32(MemoryLocation::Device, local_velocities.d_view().ptr + 0 * local_velocities.d_view().layout.stride(), flow.freestream.x, local_velocities.d_view().layout.stride());
    backend->memory->fill_f32(MemoryLocation::Device, local_velocities.d_view().ptr + 1 * local_velocities.d_view().layout.stride(), flow.freestream.y, local_velocities.d_view().layout.stride());
    backend->memory->fill_f32(MemoryLocation::Device, local_velocities.d_view().ptr + 2 * local_velocities.d_view().layout.stride(), flow.freestream.z, local_velocities.d_view().layout.stride());
    backend->rhs_assemble_velocities(rhs.d_view(), mesh.normals.d_view(), local_velocities.d_view());
    backend->lu_factor(lhs.d_view());
    backend->lu_solve(lhs.d_view(), rhs.d_view(), gamma_wing.d_view());
    backend->gamma_shed(gamma_wing.d_view(), gamma_wing_prev.d_view(), gamma_wake.d_view(), 0);
    backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());
    return AeroCoefficients{
        backend->coeff_steady_cl_multi(mesh.verts_wing.d_view(), gamma_wing_delta.d_view(), flow, mesh.area.d_view()),
        backend->coeff_steady_cd_multi(mesh.verts_wake.d_view(), gamma_wake.d_view(), flow, mesh.area.d_view()),
        {0.0f, 0.0f, 0.0f} // todo implement
    };
}

NLVLM::NLVLM(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes), 
    lhs(*backend->memory), rhs(*backend->memory), gamma_wing(*backend->memory), gamma_wake(*backend->memory), gamma_wing_prev(*backend->memory), gamma_wing_delta(*backend->memory), local_velocities(*backend->memory), strip_alphas(*backend->memory) {
    
    const u64 nw = 1;
    u64 off_wake_p = 0;
    u64 off_wake_v = 0;
    for (u32 i = 0; i < wing_panels.size(); i++) {
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
    const u64 n = wing_panels.back().offset + wing_panels.back().size();
    condition0.resize(wing_panels.size());
    lhs.alloc(Matrix<MatrixLayout::ColMajor>{n, n, n});
    rhs.alloc(MultiSurface{wing_panels, 1});
    gamma_wing.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_prev.alloc(MultiSurface{wing_panels, 1});
    gamma_wing_delta.alloc(MultiSurface{wing_panels, 1});
    gamma_wake.alloc(MultiSurface{wake_panels, 1});
    local_velocities.alloc(MultiSurface{wing_panels, 3});
    std::vector<SurfaceDims> strip_alpha_layout = wing_panels;
    for (auto& dims : strip_alpha_layout) dims.nc = 1;
    strip_alphas.alloc(MultiSurface{strip_alpha_layout, 1});
}

void strip_alpha_to_vel(const FlowData& flow, View<f32, MultiSurface>& local_velocities, const View<f32, MultiSurface>& strip_alphas) {
    for (u64 m = 0; m < local_velocities.layout.surfaces().size(); m++) {
        assert(strip_alphas.layout.ns(m) == local_velocities.layout.ns(m));
        f32* strip_alphas_m = strip_alphas.ptr + strip_alphas.layout.offset(m);
        f32* local_velocities_m = local_velocities.ptr + local_velocities.layout.offset(m);
        // todo: this can be done with fill_f32 (eliminating the need for mem transfer)
        for (u64 j = 0; j < strip_alphas.layout.ns(m); j++) {
            auto fs = compute_freestream(flow.u_inf, strip_alphas_m[j], 0.0f);
            local_velocities_m[0*local_velocities.layout.stride() + j] = fs.x;
            local_velocities_m[1*local_velocities.layout.stride() + j] = fs.y;
            local_velocities_m[2*local_velocities.layout.stride() + j] = fs.z;
        }
        for (u64 i = 1; i < local_velocities.layout.nc(m); i++) {
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
    std::fill(wing_positions.begin(), wing_positions.end(), init_pos);

    backend->memory->fill_f32(MemoryLocation::Device, lhs.d_view().ptr, 0.f, lhs.d_view().size());
    backend->memory->fill_f32(MemoryLocation::Host, strip_alphas.h_view().ptr, flow.alpha, strip_alphas.h_view().size());
    backend->memory->copy(MemoryTransfer::DeviceToDevice, mesh.verts_wing.d_view().ptr, mesh.verts_wing_init.d_view().ptr, mesh.verts_wing.d_view().size_bytes());
    backend->lu_allocate(lhs.d_view());

    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 0);
    backend->displace_wing(wing_positions, mesh.verts_wing.d_view(), mesh.verts_wing_init.d_view());
    backend->wake_shed(mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), 1);
    backend->mesh_metrics(flow.alpha, mesh.verts_wing.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.area.d_view());
    backend->lhs_assemble(lhs.d_view(), mesh.colloc.d_view(), mesh.normals.d_view(), mesh.verts_wing.d_view(), mesh.verts_wake.d_view(), condition0,1);
    backend->lu_factor(lhs.d_view());

    for (u64 iter = 0; iter < max_iter && err > tol; iter++) {
        err = 0.0; // reset l1 error
        strip_alpha_to_vel(flow, local_velocities.h_view(), strip_alphas.h_view()); // Compute local panel velocities based on strip alphas
        local_velocities.to_device();
        backend->memory->fill_f32(MemoryLocation::Device, rhs.d_view().ptr, 0.f, rhs.d_view().size());
        backend->rhs_assemble_velocities(rhs.d_view(), mesh.normals.d_view(), local_velocities.d_view());
        backend->lu_solve(lhs.d_view(), rhs.d_view(), gamma_wing.d_view());
        backend->gamma_delta(gamma_wing_delta.d_view(), gamma_wing.d_view());
 
        // Parallel Reduce
        // loop over the chordwise strips and apply Van Dam algorithm
        for (u64 m = 0; m < strip_alphas.h_view().layout.surfaces().size(); m++) {
            for (u64 j = 0; j < strip_alphas.h_view().layout.ns(m); j++) {
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
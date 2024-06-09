#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "tinycpuid.hpp"

#include <string>

#ifdef VLM_CPU
#include "vlm_backend_cpu.hpp"
#endif
#ifdef VLM_CUDA
#include "vlm_backend_cuda.hpp"
#endif

using namespace vlm;

f32 Backend::compute_coefficient_cl(const FlowData& flow) {
    return compute_coefficient_cl(flow, mesh.s_ref, 0, mesh.ns);
}

linalg::alias::float3 Backend::compute_coefficient_cm(const FlowData& flow) {
    return compute_coefficient_cm(flow, mesh.s_ref, mesh.c_ref, 0, mesh.ns);
}

f32 Backend::compute_coefficient_cd(const FlowData& flow) {
    return compute_coefficient_cd(flow, mesh.s_ref, 0, mesh.ns);
}

std::unique_ptr<Backend> vlm::create_backend(const std::string& backend_name, Mesh& mesh) {
    //tiny::CPUID cpuid;
    //cpuid.print_info();

    #ifdef VLM_CPU
    if (backend_name == "cpu") {
        return std::make_unique<BackendCPU>(mesh);
    }
    #endif
    #ifdef VLM_CUDA
    if (backend_name == "cuda") {
        return std::make_unique<BackendCUDA>(mesh);
    }
    #endif
    throw std::runtime_error("Unsupported backend: " + backend_name);
}

std::vector<std::string> vlm::get_available_backends() {
    std::vector<std::string> backends;
    #ifdef VLM_CPU
    backends.push_back("cpu");
    #endif
    #ifdef VLM_CUDA
    backends.push_back("cuda");
    #endif
    return backends;
}

Backend::Backend(MeshGeom* mesh_geom) : hh_mesh_geom(mesh_geom) {};

void Backend::init(u64 timesteps) {
    u64 nb_vertices_wing = (hh_mesh_geom->nc+1)*(hh_mesh_geom->ns+1);

    // Mesh structs allocation
    hd_mesh_geom = (MeshGeom*)allocator.h_malloc(sizeof(MeshGeom));
    dd_mesh_geom = (MeshGeom*)allocator.d_malloc(sizeof(MeshGeom));
    hh_mesh = (Mesh2*)allocator.h_malloc(sizeof(Mesh2));
    hd_mesh = (Mesh2*)allocator.h_malloc(sizeof(Mesh2));
    dd_mesh = (Mesh2*)allocator.d_malloc(sizeof(Mesh2));

    // Allocate mesh buffers on device and host
    hd_mesh_geom->vertices = (f32*) allocator.d_malloc(nb_vertices_wing*3*sizeof(f32));
    // mesh_alloc(allocator.h_malloc, hh_mesh, hh_mesh_geom->nc, hh_mesh_geom->ns, timesteps, 0);
    mesh_alloc(allocator.d_malloc, hd_mesh, hh_mesh_geom->nc, hh_mesh_geom->ns, timesteps, 0); 
    
    // Copy indices
    hd_mesh_geom->nc = hh_mesh_geom->nc;
    hd_mesh_geom->ns = hh_mesh_geom->ns;
    hd_mesh->nc = hh_mesh_geom->nc;
    hd_mesh->ns = hh_mesh_geom->ns;
    hd_mesh->nw = timesteps;

    // Copy raw vertex geometry directly to device
    allocator.hd_memcpy(hd_mesh_geom->vertices, hh_mesh_geom->vertices, nb_vertices_wing*3*sizeof(f32));
    allocator.hd_memcpy(hd_mesh->vertices, hh_mesh_geom->vertices, nb_vertices_wing*3*sizeof(f32));

    // Copy host-device mesh ptr to device-device
    allocator.hd_memcpy(dd_mesh_geom, hd_mesh_geom, sizeof(*hd_mesh_geom));
    allocator.hd_memcpy(dd_mesh, hd_mesh, sizeof(*hd_mesh));
}
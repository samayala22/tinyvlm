#pragma once

#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_backend.hpp"
#include "tinyfwd.hpp"

namespace vlm {

class Simulation {
    public:
    std::unique_ptr<Backend> backend;
    Mesh2 mesh;

    Simulation(const std::string& backend_name, const std::vector<std::string>& meshes) : backend(create_backend(backend_name)), mesh(backend->allocator) {
        // Read the sizes of all the meshes
        {
            u64 off_wing_p = 0;
            u64 off_wing_v = 0;
            for (const auto& m_name : meshes) {
                const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
                auto [nc,ns] = mesh_io.get_dims(m_name);
                mesh.params.emplace_back(MeshParams{nc, ns, 0, 0, off_wing_p, off_wing_v, 0, 0, true});
                off_wing_p += mesh.params.back().nb_panels_wing();
                off_wing_v += mesh.params.back().nb_vertices_wing();
            }
        }
        mesh.alloc_wing();

        // Perform the actual read of the mesh files
        for (u64 i = 0; i < meshes.size(); i++) {
            const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
            mesh_io.read(meshes[i], mesh.verts_wing_init.h_ptr() + mesh.params[i].off_wing_p);
        }
    };
    virtual ~Simulation() = default;
};

class VLM final: public Simulation {
    public:
        VLM(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~VLM() = default;
        AeroCoefficients run(const FlowData& flow);
    private:
        void alloc_buffers();
        Buffer<f32, Device> lhs; // (ns*nc)^2
        Buffer<f32, Device> rhs; // ns*nc
        Buffer<f32, HostDevice> gamma_wing; // nc*ns
        Buffer<f32, Device> gamma_wake; // nw*ns
        Buffer<f32, Device> gamma_wing_prev; // nc*ns
        Buffer<f32, Device> gamma_wing_delta; // nc*ns
        Buffer<f32, Device> local_velocities; // ns*nc*3
};

// class Solver {
//     public:
//     std::unique_ptr<Backend> backend;
//     Solver(const tiny::Config& cfg);
//     Solver(std::unique_ptr<Backend>&& backend_) : backend(std::move(backend_)) {};
//     virtual ~Solver() = default;
// };

// class NonLinearVLM final: public Solver {
//     public:
//     static constexpr f64 DEFAULT_TOL = 1e-5;
//     static constexpr u64 DEFAULT_MAX_ITER = 100;

//     const f64 tol;
//     const u64 max_iter;
//     f32* strip_alphas = nullptr; // ns
//     f32* velocities = nullptr; // nc*ns*3

//     NonLinearVLM( // init from cfg
//         const tiny::Config& cfg,
//         const f64 tol_= DEFAULT_TOL,
//         const u64 max_iter_ = DEFAULT_MAX_ITER
//     ):
//         Solver(cfg),
//         tol(tol_),
//         max_iter(max_iter_)
//     {
//         alloc();
//     } 
//     NonLinearVLM( // init from backend
//         std::unique_ptr<Backend>&& backend_,
//         const f64 tol_= DEFAULT_TOL,
//         const u64 max_iter_ = DEFAULT_MAX_ITER
//     ) :
//         Solver(std::move(backend_)),
//         tol(tol_),
//         max_iter(max_iter_)
//     {
//         alloc();
//     }
//     ~NonLinearVLM() {
//         dealloc();
//     };
//     AeroCoefficients solve(const FlowData& flow, const Database& db);

//     private:
//     void alloc() {
//         strip_alphas = (f32*)backend->allocator.h_malloc(backend->hd_mesh->ns * sizeof(f32));
//         velocities = (f32*)backend->allocator.h_malloc(backend->hd_mesh->nc * backend->hd_mesh->ns * 3 * sizeof(f32));
//     }
//     void dealloc() {
//         backend->allocator.h_free(strip_alphas);
//         backend->allocator.h_free(velocities);        
//     }
// };

// class LinearVLM final: public Solver {
//     public:
//     LinearVLM(const tiny::Config& cfg): Solver(cfg) {}
//     LinearVLM(std::unique_ptr<Backend>&& backend_): Solver(std::move(backend_)) {};
//     ~LinearVLM() = default;
//     AeroCoefficients solve(const FlowData& flow);
// };


// class UVLM final: public Solver {
//     public:

//     UVLM(const tiny::Config& cfg): Solver(cfg) {}
//     UVLM(std::unique_ptr<Backend>&& backend_): Solver(std::move(backend_)) {};
//     ~UVLM() = default;
//     AeroCoefficients solve(const FlowData& flow);
// };

} // namespace vlm

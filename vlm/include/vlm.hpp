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
        std::vector<linalg::alias::float4x4> wing_positions; // todo: move this somewhere

        Simulation(const std::string& backend_name, const std::vector<std::string>& meshes);
        virtual ~Simulation() = default;
};

class VLM final: public Simulation {
    public:
        VLM(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~VLM() = default;
        AeroCoefficients run(const FlowData& flow);
        
        // Pulic for output purposes (eg dump gamma to file)
        Buffer<f32, MemoryLocation::Device> lhs; // (ns*nc)^2
        Buffer<f32, MemoryLocation::Device> rhs; // ns*nc
        Buffer<f32, MemoryLocation::HostDevice> gamma_wing; // nc*ns
        Buffer<f32, MemoryLocation::Device> gamma_wake; // nw*ns
        Buffer<f32, MemoryLocation::Device> gamma_wing_prev; // nc*ns
        Buffer<f32, MemoryLocation::Device> gamma_wing_delta; // nc*ns
        Buffer<f32, MemoryLocation::Device> local_velocities; // ns*nc*3
    private:
        void alloc_buffers();
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
//         strip_alphas = (f32*)backend->memory.h_malloc(backend->hd_mesh->ns * sizeof(f32));
//         velocities = (f32*)backend->memory.h_malloc(backend->hd_mesh->nc * backend->hd_mesh->ns * 3 * sizeof(f32));
//     }
//     void dealloc() {
//         backend->memory.h_free(strip_alphas);
//         backend->memory.h_free(velocities);        
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

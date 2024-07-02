#pragma once

#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_backend.hpp"
#include "tinyfwd.hpp"

namespace vlm {

class Solver {
    public:
    std::unique_ptr<Backend> backend;
    Solver(const tiny::Config& cfg);
    Solver(std::unique_ptr<Backend>&& backend_) : backend(std::move(backend_)) {};
    virtual ~Solver() = default;
};

class NonLinearVLM final: public Solver {
    public:
    static constexpr f64 DEFAULT_TOL = 1e-5;
    static constexpr u64 DEFAULT_MAX_ITER = 100;

    const f64 tol;
    const u64 max_iter;
    f32* strip_alphas = nullptr; // ns
    f32* velocities = nullptr; // nc*ns*3

    NonLinearVLM( // init from cfg
        const tiny::Config& cfg,
        const f64 tol_= DEFAULT_TOL,
        const u64 max_iter_ = DEFAULT_MAX_ITER
    ):
        Solver(cfg),
        tol(tol_),
        max_iter(max_iter_)
    {
        alloc();
    } 
    NonLinearVLM( // init from backend
        std::unique_ptr<Backend>&& backend_,
        const f64 tol_= DEFAULT_TOL,
        const u64 max_iter_ = DEFAULT_MAX_ITER
    ) :
        Solver(std::move(backend_)),
        tol(tol_),
        max_iter(max_iter_)
    {
        alloc();
    }
    ~NonLinearVLM() {
        dealloc();
    };
    AeroCoefficients solve(const FlowData& flow, const Database& db);

    private:
    void alloc() {
        strip_alphas = (f32*)backend->allocator.h_malloc(backend->hd_mesh->ns * sizeof(f32));
        velocities = (f32*)backend->allocator.h_malloc(backend->hd_mesh->nc * backend->hd_mesh->ns * 3 * sizeof(f32));
    }
    void dealloc() {
        backend->allocator.h_free(strip_alphas);
        backend->allocator.h_free(velocities);        
    }
};

class LinearVLM final: public Solver {
    public:
    LinearVLM(const tiny::Config& cfg): Solver(cfg) {}
    LinearVLM(std::unique_ptr<Backend>&& backend_): Solver(std::move(backend_)) {};
    ~LinearVLM() = default;
    AeroCoefficients solve(const FlowData& flow);
};

class UVLM final: public Solver {
    public:

    UVLM(const tiny::Config& cfg): Solver(cfg) {}
    UVLM(std::unique_ptr<Backend>&& backend_): Solver(std::move(backend_)) {};
    ~UVLM() = default;
    AeroCoefficients solve(const FlowData& flow);
};

} // namespace vlm

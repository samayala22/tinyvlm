#pragma once


#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class Solver {
    public:
    Mesh mesh; // not const because we have to change the wake vertices according to alpha
    std::unique_ptr<Backend> backend;
    Solver(const tiny::Config& cfg); // mesh filename & backend name
    ~Solver() = default;
};

// class NonLinearVLM final: public Solver {
//     public:
//     const f64 tol;
//     const u32 max_iter;
//     std::vector<f32> strip_alphas;

//     NonLinearVLM(
//         const tiny::Config& cfg,
//         const f64 tol_= 1e-4f,
//         const u32 max_iter_ = 100
//     ):
//         Solver(cfg),
//         tol(tol_),
//         max_iter(max_iter_)
//     {}
//     ~NonLinearVLM() = default;
//     void solve(const FlowData& flow, const Database& db);
// };

class LinearVLM final: public Solver {
    public:
    LinearVLM(const tiny::Config& cfg): Solver(cfg) {}
    ~LinearVLM() = default;
    void solve(const FlowData& flow);
};

} // namespace vlm

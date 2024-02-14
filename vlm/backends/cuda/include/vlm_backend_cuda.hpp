#pragma once 

#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"

#include "vlm_backend_cpu.hpp" // temporary

namespace vlm {

class BackendCUDA : public Backend {
    public:
        BackendCPU default_backend; // temporary

        float* d_lhs = nullptr;
        float* d_rhs = nullptr;
        float* d_gamma = nullptr;
        float* d_delta_gamma = nullptr;

        BackendCUDA(Mesh& mesh);
        ~BackendCUDA();

        void reset() override;
        void compute_lhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) override; 
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;
        f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
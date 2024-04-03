#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendCPU : public Backend {
    public:
        std::vector<f32> lhs;
        std::vector<f32> wake_buffer; // (ns*nc) * ns
        std::vector<f32> rhs;
        std::vector<i32> ipiv;
        std::vector<f32> gamma; // ncw * ns
        std::vector<f32> delta_gamma;
        std::vector<f32> trefftz_buffer;

        BackendCPU(Mesh& mesh);
        ~BackendCPU();
        void reset() override;
        void compute_lhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) override; 
        void add_wake_influence(const FlowData& flow) override;
        void shed_gamma() override;
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;
        f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendAVX2 : public Backend {
    public:
        struct linear_solver_t;
        std::unique_ptr<linear_solver_t> solver;

        std::vector<f32> lhs;
        std::vector<f32> rhs;
        std::vector<f32> gamma;
        std::vector<f32> delta_gamma;

        BackendAVX2(Mesh& mesh);
        ~BackendAVX2();
        void reset() override;
        void compute_lhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) override; 
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u32 j, const u32 n) override;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u32 j, const u32 n) override;
        f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u32 j, const u32 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
#pragma once 

#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"

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
        ~BackendAVX2() = default;
        void reset() override;
        void compute_lhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) override; 
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u32 j, const u32 n) override;
        Eigen::Vector3f compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u32 j, const u32 n) override;
        f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u32 j, const u32 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
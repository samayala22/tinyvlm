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

        BackendAVX2(Mesh& mesh, Data& data);
        ~BackendAVX2() = default;
        void reset() override;
        void compute_lhs() override;
        void compute_rhs() override;
        void compute_rhs(const std::vector<f32>& section_alphas) override; 
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const Mesh& mesh, const Data& data, const f32 area, const Eigen::Vector3f& freestream, const u32 j, const u32 n) override;
        Eigen::Vector3f compute_coefficient_cm(const Mesh& mesh, const Data& data, const f32 area, const f32 chord, const u32 j, const u32 n) override;
        f32 compute_coefficient_cd(const Mesh& mesh, const Data& data, const f32 area, const u32 j, const u32 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
#pragma once 

#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"

#include "vlm_backend_avx2.hpp" // temporary

namespace vlm {

class BackendCUDA : public Backend {
    public:
        BackendAVX2 default_backend; // temporary

        float* d_lhs = nullptr;
        float* d_rhs = nullptr;

        BackendCUDA(Mesh& mesh, Data& data);
        ~BackendCUDA();

        void reset() override;
        void compute_lhs() override;
        void compute_rhs() override;
        void rebuild_rhs(const std::vector<f32>& section_alphas) override; 
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const Mesh& mesh, const Data& data, const f32 area, const u32 j, const u32 n) override;
        Eigen::Vector3f compute_coefficient_cm(const Mesh& mesh, const Data& data, const f32 area, const f32 chord, const u32 j, const u32 n) override;
        f32 compute_coefficient_cd(const Mesh& mesh, const Data& data, const f32 area, const u32 j, const u32 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
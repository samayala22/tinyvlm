#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendCPU : public Backend {
    public:
        std::vector<f32> lhs;
        std::vector<f32> uw;
        std::vector<f32> vw;
        std::vector<f32> ww;
        std::vector<f32> panel_uw;
        std::vector<f32> panel_vw;
        std::vector<f32> panel_ww;
        SoA_3D_t<f32> rollup_vertices;
        std::vector<f32> wake_buffer; // (ns*nc) * ns
        std::vector<f32> rhs;
        std::vector<i32> ipiv;
        std::vector<f32> gamma; // ncw * ns
        std::vector<f32> gamma_prev; // nc * ns (previous timestep gamma)
        std::vector<f32> delta_gamma;
        std::vector<f32> trefftz_buffer;

        BackendCPU(Mesh& mesh);
        ~BackendCPU();
        void reset() override;
        void compute_lhs() override;
        void compute_rhs(const FlowData& flow) override;
        void compute_rhs(const SoA_3D_t<f32>& velocities) override; 
        void add_wake_influence() override;
        void wake_rollup(float dt) override;
        void shed_gamma() override;
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        f32 compute_coefficient_unsteady_cl(const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) override;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;
        f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
#pragma once 

#include "vlm_fwd.hpp"
#include "vlm_backend.hpp"

namespace vlm {

class BackendCPU : public Backend {
    public:
        BackendCPU();
        ~BackendCPU();
        void reset() override;
        void lhs_assemble() override;
        void compute_rhs() override;
        void add_wake_influence() override;
        void wake_rollup(float dt) override;
        void shed_gamma() override;
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        f32 compute_coefficient_unsteady_cl(const linalg::alias::float3& freestream, const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) override;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;
        f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        void compute_delta_gamma() override;
        void set_velocities(const linalg::alias::float3& vel) override;
        void set_velocities(const SoA_3D_t<f32>& vels) override;
};

} // namespace vlm
#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_mesh.hpp"

namespace vlm {

class Backend {
    public:
        Mesh& mesh;
        f32 sigma_vatistas = 0.0f;

        Backend(Mesh& mesh) : mesh(mesh) {};
        virtual void reset() = 0;
        virtual void lhs_assemble() = 0;
        virtual void compute_rhs() = 0;
        virtual void add_wake_influence() = 0;
        virtual void wake_rollup(float dt) = 0;
        virtual void shed_gamma() = 0;
        virtual void lu_factor() = 0;
        virtual void lu_solve() = 0;
        virtual f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;
        virtual f32 compute_coefficient_unsteady_cl(const linalg::alias::float3& freestream, const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) = 0;
        f32 compute_coefficient_cl(const FlowData& flow);
        virtual linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) = 0;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow);
        virtual f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;
        f32 compute_coefficient_cd(const FlowData& flow);
        virtual void compute_delta_gamma() = 0;
        virtual void set_velocities(const linalg::alias::float3& vel) = 0;
        virtual void set_velocities(const SoA_3D_t<f32>& vels) = 0;
        virtual ~Backend() = default;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name, Mesh& mesh);
std::vector<std::string> get_available_backends();

} // namespace vlm

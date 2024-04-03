#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"

namespace vlm {

class Backend {
    public:
        Mesh& mesh;

        Backend(Mesh& mesh) : mesh(mesh) {};
        virtual void reset() = 0;
        virtual void compute_lhs(const FlowData& flow) = 0;
        virtual void compute_rhs(const FlowData& flow) = 0;
        virtual void compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) = 0; 
        virtual void add_wake_influence(const FlowData& flow) = 0;
        virtual void shed_gamma() = 0;
        virtual void lu_factor() = 0;
        virtual void lu_solve() = 0;
        virtual f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;
        f32 compute_coefficient_cl(const FlowData& flow);
        virtual linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) = 0;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow);
        virtual f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;
        f32 compute_coefficient_cd(const FlowData& flow);
        virtual void compute_delta_gamma() = 0;
        virtual ~Backend() = default;
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name, Mesh& mesh);
std::vector<std::string> get_available_backends();

} // namespace vlm

#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"

namespace vlm {

class Backend {
    public:
        Mesh& mesh;
        Data& data;

        Backend(Mesh& mesh, Data& data) : mesh(mesh), data(data) {};
        virtual void reset() = 0;
        virtual void compute_lhs() = 0;
        virtual void compute_rhs() = 0;
        virtual void rebuild_rhs(const std::vector<f32>& section_alphas) = 0; 
        virtual void lu_factor() = 0;
        virtual void lu_solve() = 0;
        virtual void compute_coefficients() = 0;
        virtual f32 compute_coefficient_cl(const Mesh& mesh, const Data& data, const u32 j, const u32 n, const f32 area) = 0;
        virtual Eigen::Vector3f compute_coefficient_cm(const Mesh& mesh, const Data& data, const u32 j, const u32 n, const f32 area, const f32 chord) = 0;
        virtual f32 compute_coefficient_cd(const Mesh& mesh, const Data& data, const u32 j, const u32 n, const f32 area) = 0;
        virtual void compute_delta_gamma() = 0;
        virtual ~Backend() = default;
};

}

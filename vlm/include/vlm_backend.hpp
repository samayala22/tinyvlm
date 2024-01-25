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
        virtual void compute_rhs(const std::vector<f32>& section_alphas) = 0; 
        virtual void lu_factor() = 0;
        virtual void lu_solve() = 0;
        virtual f32 compute_coefficient_cl(const Mesh& mesh, const Data& data, const f32 area, const Eigen::Vector3f& freestream, const u32 j, const u32 n) = 0;
        virtual Eigen::Vector3f compute_coefficient_cm(const Mesh& mesh, const Data& data, const f32 area, const f32 chord, const u32 j, const u32 n) = 0;
        virtual f32 compute_coefficient_cd(const Mesh& mesh, const Data& data, const f32 area, const u32 j, const u32 n) = 0;
        f32 compute_coefficient_cl(const Mesh& mesh, const Data& data, const f32 area);
        Eigen::Vector3f compute_coefficient_cm(const Mesh& mesh, const Data& data, const f32 area, const f32 chord);
        f32 compute_coefficient_cd(const Mesh& mesh, const Data& data, const f32 area);
        virtual void compute_delta_gamma() = 0;
        virtual ~Backend() = default;
};

}
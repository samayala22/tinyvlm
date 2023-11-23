#pragma once 

#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"

namespace vlm {

class BackendAVX2 : public Backend {
    public:
        std::vector<f32> lhs;
        std::vector<f32> rhs;

        BackendAVX2(Mesh& mesh, Data& data);
        ~BackendAVX2() = default;
        void reset() override;
        void compute_lhs() override;
        void compute_rhs() override;
        void solve() override;
        void compute_forces() override;
        void compute_delta_gamma() override;

        
};

} // namespace vlm
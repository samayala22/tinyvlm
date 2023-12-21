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
        void solve() override;
        void compute_forces() override;
        void compute_delta_gamma() override;
};

} // namespace vlm
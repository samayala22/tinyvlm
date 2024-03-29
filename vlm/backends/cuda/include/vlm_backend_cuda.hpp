#pragma once 

#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"

#include "vlm_backend_cpu.hpp" // temporary

namespace vlm {

struct SoA3D {
    float* x;
    float* y;
    float* z;
};

struct MeshProxy {
    uint64_t ns;
    uint64_t nc;
    uint64_t nb_panels;
    SoA3D v; // vertices
    SoA3D colloc; // collocation points
    SoA3D normal; // normals
};

class BackendCUDA : public Backend {
    public:
        BackendCPU default_backend; // temporary

        float* d_lhs = nullptr;
        float* d_rhs = nullptr;
        float* d_gamma = nullptr;
        float* d_delta_gamma = nullptr;

        int* d_solver_info = nullptr;
        int* d_solver_ipiv = nullptr;
        float* d_solver_buffer = nullptr;
        
        MeshProxy h_mesh;
        MeshProxy* d_mesh;

        BackendCUDA(Mesh& mesh);
        ~BackendCUDA();

        void reset() override;
        void compute_lhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow) override;
        void compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) override; 
        void lu_factor() override;
        void lu_solve() override;
        f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) override;
        f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) override;
        void compute_delta_gamma() override;
};

} // namespace vlm
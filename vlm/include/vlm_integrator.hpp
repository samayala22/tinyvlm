#include "vlm_backend.hpp"

namespace vlm {

class NewmarkBeta {
    public:
        NewmarkBeta(Backend* backend, const f32 beta = 0.25f, const f32 gamma = 0.5f) : 
            m_memory(backend->create_memory_manager()),
            m_blas(backend->create_blas()),
            m_solver(backend->create_lu_solver()),
            m_beta(beta),
            m_gamma(gamma) {};
        ~NewmarkBeta() = default;

        void init(
            const TensorView2fD& M,
            const TensorView2fD& C,
            const TensorView2fD& K, f32 dt);

        void step(
            const TensorView2fD& M,
            const TensorView2fD& C,
            const TensorView1fD& v_i,
            const TensorView1fD& a_i,
            const TensorView1fD& du,
            const TensorView1fD& dv,
            const TensorView1fD& da,
            const TensorView1fD& delta_F,
            const f32 dt);
    private:
        std::unique_ptr<Memory> m_memory;
        std::unique_ptr<BLAS> m_blas;
        std::unique_ptr<LU> m_solver;
        
        Tensor<f32, 2, Location::Device> m_K_eff{m_memory.get()};
        Tensor<f32, 1, Location::Device> m_factor{m_memory.get()};
        
        const f32 m_beta;
        const f32 m_gamma;
};

} // namespace vlm
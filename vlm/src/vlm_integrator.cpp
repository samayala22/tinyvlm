#include "vlm_integrator.hpp"

using namespace vlm;

void NewmarkBeta::init(
    const TensorView<f32, 2, Location::Device>& M,
    const TensorView<f32, 2, Location::Device>& C,
    const TensorView<f32, 2, Location::Device>& K,
    const f32 dt) 
{
    m_K_eff.init({M.shape(0), M.shape(1)});
    m_factor.init({M.shape(0)});

    auto K_eff = m_K_eff.view();
    m_solver->init(K_eff);

    const f32 x1 = m_gamma / (m_beta * dt);
    const f32 x0 = 1 / (m_beta * dt*dt);
    
    K.to(K_eff);
    m_blas->axpy(x0, M, K_eff);
    m_blas->axpy(x1, C, K_eff);
    m_solver->factorize(K_eff);
}

void NewmarkBeta::step(
    const TensorView<f32, 2, Location::Device>& M,
    const TensorView<f32, 2, Location::Device>& C,
    const TensorView<f32, 1, Location::Device>& v_i,
    const TensorView<f32, 1, Location::Device>& a_i,
    const TensorView<f32, 1, Location::Device>& du,
    const TensorView<f32, 1, Location::Device>& dv,
    const TensorView<f32, 1, Location::Device>& da,
    const TensorView<f32, 1, Location::Device>& delta_F,
    const f32 dt) 
{

    const f32 x1 = m_gamma / (m_beta * dt);
    const f32 x0 = 1 / (m_beta * dt*dt);
    const f32 xd0 = 1 / (m_beta * dt);
    const f32 xd1 = m_gamma / m_beta;
    const f32 xdd0 = 1/(2*m_beta);
    const f32 xdd1 = - dt * (1 - m_gamma / (2*m_beta));

    auto K_eff = m_K_eff.view();
    auto factor = m_factor.view();

    delta_F.to(du);
    factor.fill(0.f);
    m_blas->axpy(xd0, v_i, factor);
    m_blas->axpy(xdd0, a_i, factor);
    m_blas->gemv(1.0f, M, factor, 1.0f, du);
    factor.fill(0.f);
    m_blas->axpy(xd1, v_i, factor);
    m_blas->axpy(xdd1, a_i, factor);
    m_blas->gemv(1.0f, C, factor, 1.0f, du);

    m_solver->solve(K_eff, du.reshape(du.size(), 1));
    
    dv.fill(0.f);
    m_blas->axpy(x1, du, dv);
    m_blas->axpy(-xd1, v_i, dv);
    m_blas->axpy(-xdd1, a_i, dv);
    da.fill(0.f);
    m_blas->axpy(x0, du, da);
    m_blas->axpy(-xd0, v_i, da);
    m_blas->axpy(-xdd0, a_i, da);
}

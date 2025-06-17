#include "vlm_solvers.hpp"
#include "vlm_backend.hpp"

using namespace vlm;

template<typename T>
void anderson_acceleration_impl(
    Backend* backend,
    const TensorView<T, 1, Location::Device>& x0,
    const std::function<void(const TensorView<T, 1, Location::Device>& x, const TensorView<T, 1, Location::Device>& y)>& f,
    i32 max_iter = 100,
    T tol_res = 1e-6,
    i32 m = 3
)
{
    // tiny::ScopedTimer timer("anderson_acceleration");
    i64 n = x0.shape(0);
    Tensor<T, 2, Location::Device> m_X_buf{backend->memory.get()};
    Tensor<T, 2, Location::Device> m_G_buf{backend->memory.get()};
    Tensor<T, 2, Location::Device> m_G_buf_k{backend->memory.get()}; // copy because lsq overwrites
    
    Tensor<T, 1, Location::Device> m_x_curr{backend->memory.get()};
    Tensor<T, 1, Location::Device> m_x_new{backend->memory.get()};
    Tensor<T, 1, Location::Device> m_g_curr{backend->memory.get()};
    Tensor<T, 1, Location::Device> m_g_new{backend->memory.get()};
    Tensor<T, 1, Location::Device> m_gamma{backend->memory.get()};
    auto lsq_solver = backend->create_lsq_solver();

    m_X_buf.init({n, m});
    m_G_buf.init({n, m});
    m_G_buf_k.init({n, m});
    m_x_curr.init({n});
    m_x_new.init({n});
    m_g_curr.init({n});
    m_g_new.init({n});
    m_gamma.init({n});

    auto& X_buf = m_X_buf.view();
    auto& G_buf = m_G_buf.view();
    auto& G_buf_k = m_G_buf_k.view();
    auto& x_curr = m_x_curr.view();
    auto& x_new = m_x_new.view();
    auto& g_curr = m_g_curr.view();
    auto& g_new = m_g_new.view();
    auto& gamma = m_gamma.view();

    lsq_solver->init(G_buf_k, gamma.reshape(gamma.shape(0), 1));

    X_buf.fill(0.f); // not necessary
    G_buf.fill(0.f); // not necessary
    G_buf_k.fill(0.f); // not necessary

    x0.to(x_curr);
    f(x_curr, x_new);
    x_new.to(g_curr);
    backend->blas->axpy(-1.0f, x_curr, g_curr);
    g_curr.to(X_buf.slice(All, 0));
    
    x_new.to(x_curr);
    f(x_curr, x_new);
    x_new.to(g_new);
    backend->blas->axpy(-1.0f, x_curr, g_new);
    g_new.to(G_buf.slice(All, 0));
    backend->blas->axpy(-1.0f, g_curr, G_buf.slice(All, 0));
    g_new.to(g_curr);

    i32 k = 1;
    while (k < max_iter && backend->blas->norm(g_curr) > tol_res) {
        i32 m_k = std::min(m, k);
        
        g_curr.to(gamma);
        auto G_bufk = G_buf.slice(All, Range{0, m_k});
        auto G_bufk_k = G_buf_k.slice(All, Range{0, m_k});
        auto X_bufk = X_buf.slice(All, Range{0, m_k});
        auto gammak = gamma.slice(Range{0, m_k});

        G_bufk.to(G_bufk_k);
        lsq_solver->solve(G_bufk_k, gamma.reshape(gamma.shape(0), 1));
        x_curr.to(x_new); // i think not necessary
        backend->blas->axpy(1.0f, g_curr, x_new);
        backend->blas->gemv(-1.0f, X_bufk, gammak, 1.0f, x_new);
        backend->blas->gemv(-1.0f, G_bufk, gammak, 1.0f, x_new);
        
        // i32 m_bk = std::min(m_k, m-1);
        // if (m_k == m) {
        //     for (i32 i = 0; i < m-1; i++) {
        //         X_bufk.slice(All, i+1).to(X_bufk.slice(All, i));
        //         G_bufk.slice(All, i+1).to(G_bufk.slice(All, i));
        //     }
        // }
        // auto X_bufbk = X_buf.slice(All, m_bk);
        // auto G_bufbk = G_buf.slice(All, m_bk);

        auto X_bufbk = X_buf.slice(All, k % m);
        auto G_bufbk = G_buf.slice(All, k % m);

        x_new.to(X_bufbk);
        backend->blas->axpy(-1.0f, x_curr, X_bufbk);
        x_new.to(x_curr);
        f(x_curr, x_new);
        x_new.to(g_new);
        backend->blas->axpy(-1.0f, x_curr, g_new);
        g_new.to(G_bufbk);
        backend->blas->axpy(-1.0f, g_curr, G_bufbk);
        g_new.to(g_curr);
        k += 1;
    }
    x_curr.to(x0);

    if (k == max_iter) {
        std::printf("Anderson method failed to converge");
    } else {
        std::printf("Anderson fixed point converged in %d iterations\n", k);
    }
}

void vlm::anderson_acceleration(
    Backend* backend,
    const TensorView1dD& x0,
    const std::function<void(const TensorView1dD& x, const TensorView1dD& y)>& f,
    i32 max_iter,
    f64 tol_res,
    i32 m
) {
    anderson_acceleration_impl<f64>(
        backend,
        x0,
        f,
        max_iter,
        tol_res,
        m
    );
}

void vlm::anderson_acceleration(
    Backend* backend,
    const TensorView1fD& x0,
    const std::function<void(const TensorView1fD& x, const TensorView1fD& y)>& f,
    i32 max_iter,
    f32 tol_res,
    i32 m
) {
    anderson_acceleration_impl<f32>(
        backend,
        x0,
        f,
        max_iter,
        tol_res,
        m
    );
}
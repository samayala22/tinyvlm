#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"

#include "tinycombination.hpp"
#include "tinypbar.hpp"

#include <tuple>
#include <fstream>
#include <string>

using namespace vlm;
 
class NewmarkBeta {
    public:
        NewmarkBeta(Backend* backend, const f32 beta = 0.25f, const f32 gamma = 0.5f) : 
            m_memory(backend->create_memory_manager()),
            m_blas(backend->create_blas()),
            m_solver(backend->create_lu_solver()),
            m_beta(beta),
            m_gamma(gamma) {};
        ~NewmarkBeta() = default;

        void init(TensorView<f32, 2, Location::Device>& M,
            TensorView<f32, 2, Location::Device>& C,
            TensorView<f32, 2, Location::Device>& K, f32 dt);

        void step(
            TensorView<f32, 2, Location::Device>& M,
            TensorView<f32, 2, Location::Device>& C,
            TensorView<f32, 1, Location::Device>& v_i,
            TensorView<f32, 1, Location::Device>& a_i,
            TensorView<f32, 1, Location::Device>& du,
            TensorView<f32, 1, Location::Device>& dv,
            TensorView<f32, 1, Location::Device>& da,
            TensorView<f32, 1, Location::Device>& delta_F,
            const f32 dt);
    private:
        std::unique_ptr<Memory> m_memory;
        std::unique_ptr<BLAS> m_blas;
        std::unique_ptr<LU> m_solver;
        
        Tensor<f32, 2, Location::Device> m_K_eff{*m_memory};
        Tensor<f32, 1, Location::Device> m_factor{*m_memory};
        
        const f32 m_beta;
        const f32 m_gamma;
};

void NewmarkBeta::init(TensorView<f32, 2, Location::Device>& M,
    TensorView<f32, 2, Location::Device>& C,
    TensorView<f32, 2, Location::Device>& K, f32 dt) 
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
    TensorView<f32, 2, Location::Device>& M,
    TensorView<f32, 2, Location::Device>& C,
    TensorView<f32, 1, Location::Device>& v_i,
    TensorView<f32, 1, Location::Device>& a_i,
    TensorView<f32, 1, Location::Device>& du,
    TensorView<f32, 1, Location::Device>& dv,
    TensorView<f32, 1, Location::Device>& da,
    TensorView<f32, 1, Location::Device>& delta_F,
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

int main() {
    // vlm::Executor::instance(1);
    const std::vector<std::string> backends = get_available_backends();
    auto simulations = tiny::make_combination(backends);

    f32 dt = 0.1;
    f32 t_final = 20;
    i64 tsteps = static_cast<i64>(t_final / dt) + 1;
    
    for (const auto& [backend_name] : simulations) {
        // UVLM_2DOF simulation{backend_name, {mesh_name}};
        // simulation.run({kinematics}, {initial_pose}, t_final);
        std::unique_ptr<Backend> backend = create_backend(backend_name);
        std::unique_ptr<Memory> memory = backend->create_memory_manager();
        std::unique_ptr<BLAS> blas = backend->create_blas();
        std::unique_ptr<LU> solver = backend->create_lu_solver();
        
        Tensor2D<Location::Host> M_h{*memory};
        Tensor2D<Location::Host> C_h{*memory};
        Tensor2D<Location::Host> K_h{*memory};
        Tensor2D<Location::Host> u_h{*memory};
        Tensor2D<Location::Host> v_h{*memory};
        Tensor2D<Location::Host> a_h{*memory};
        Tensor1D<Location::Host> t_h{*memory};

        Tensor1D<Location::Device> zero{*memory};
        Tensor2D<Location::Device> M_d{*memory};
        Tensor2D<Location::Device> C_d{*memory};
        Tensor2D<Location::Device> K_d{*memory};
        Tensor2D<Location::Device> u_d{*memory}; // dof x tsteps
        Tensor2D<Location::Device> v_d{*memory}; // dof x tsteps
        Tensor2D<Location::Device> a_d{*memory}; // dof x tsteps
        Tensor1D<Location::Device> du{*memory}; // dof
        Tensor1D<Location::Device> dv{*memory}; // dof
        Tensor1D<Location::Device> da{*memory}; // dof

        // Host tensors
        M_h.init({3,3});
        C_h.init({3,3});
        K_h.init({3,3});
        u_h.init({3, tsteps});
        v_h.init({3, tsteps});
        a_h.init({3, tsteps});
        t_h.init({tsteps});

        // Device tensors
        zero.init({3});
        M_d.init({3,3});
        C_d.init({3,3});
        K_d.init({3,3});
        u_d.init({3,tsteps});
        v_d.init({3,tsteps});
        a_d.init({3,tsteps});
        du.init({3});
        dv.init({3});
        da.init({3});

        // System matrices
        K_h[0] = 400.0f;
        K_h[1] = -200.0f;
        K_h[2] = 0.0f;
        K_h[3] = -200.0f;
        K_h[4] = 400.0f;
        K_h[5] = -200.0f;
        K_h[6] = 0.0f;
        K_h[7] = -200.0f;
        K_h[8] = 200.0f;

        C_h[0] = 0.55f;
        C_h[1] = -0.2f;
        C_h[2] = 0.0f;
        C_h[3] = -0.2f;
        C_h[4] = 0.4f;
        C_h[5] = -0.2f;
        C_h[6] = 0.0f;
        C_h[7] = -0.2f;
        C_h[8] = 0.35f;

        M_h.view().fill(0.0f);
        M_h[0] = 1.0f;
        M_h[4] = 1.0f;
        M_h[8] = 1.0f;
    
        // Initial condition
        t_h.view().fill(0.0f);
        auto u0_h = u_h.view().slice(All, 0);
        auto v0_h = v_h.view().slice(All, 0);
        u0_h.fill(0.0f);
        v0_h(0) = 1.0f;
        v0_h(1) = 1.0f;
        v0_h(2) = 1.0f;
        u_d.view().fill(0.0f);
        v_d.view().fill(0.0f);
        a_d.view().fill(0.0f);

        // Transfer to device
        M_h.view().to(M_d.view());
        C_h.view().to(C_d.view());
        K_h.view().to(K_d.view());
        auto a0 = a_d.view().slice(All, 0);
        auto v0 = v_d.view().slice(All, 0);
        auto u0 = u_d.view().slice(All, 0);

        // Pre-compute constant values
        u0_h.to(u0);
        v0_h.to(v0);
        zero.view().to(a0); // F0 = 0
        blas->gemv(-1.0f, C_d.view(), v0, 1.0f, a0);
        blas->gemv(-1.0f, K_d.view(), u0, 1.0f, a0);
        solver->init(M_d.view());
        solver->factorize(M_d.view());
        solver->solve(M_d.view(), a0.reshape(a0.size(), 1)); // maybe override solve for 1D rhs

        // Initialize integrator with constant dt
        NewmarkBeta integrator{backend.get()};
        integrator.init(M_d.view(), C_d.view(), K_d.view(), dt);
        
        // Integration loop
        for (i64 step = 0; step < tsteps - 1; step++) {
            t_h.view()(step+1) = static_cast<f32>(step+1) * dt;
            auto u_i = u_d.view().slice(All, step);
            auto v_i = v_d.view().slice(All, step);
            auto a_i = a_d.view().slice(All, step);

            auto u_ip1 = u_d.view().slice(All, step+1);
            auto v_ip1 = v_d.view().slice(All, step+1);
            auto a_ip1 = a_d.view().slice(All, step+1);

            integrator.step(
                M_d.view(),
                C_d.view(),
                v_i,
                a_i,
                du.view(),
                dv.view(),
                da.view(),
                zero.view(),
                dt
            );
            u_i.to(u_ip1);
            v_i.to(v_ip1);
            a_i.to(a_ip1);
            blas->axpy(1.0f, du.view(), u_ip1);
            blas->axpy(1.0f, dv.view(), v_ip1);
            blas->axpy(1.0f, da.view(), a_ip1);
        }

        // Transfer back to host
        u_d.view().to(u_h.view());
        v_d.view().to(v_h.view());
        a_d.view().to(a_h.view());
        
        // Output
        std::ios::sync_with_stdio(false);
        std::ofstream accel_data("newmark_3dof_" + backend_name.get() + ".txt");
        accel_data << 3 << " " << tsteps << "\n";
        for (i64 i = 0; i < tsteps; i++) {
            auto& u_hv = u_h.view();
            auto& v_hv = v_h.view();
            auto& a_hv = a_h.view();
            auto& t_hv = t_h.view();
            accel_data << t_hv(i) << " " \
                << u_hv(0, i) << " " << u_hv(1, i) << " " << u_hv(2, i) << " " \
                << v_hv(0, i) << " " << v_hv(1, i) << " " << v_hv(2, i) << " " \
                << a_hv(0, i) << " " << a_hv(1, i) << " " << a_hv(2, i) << "\n";
        }
    }
    return 0;
}
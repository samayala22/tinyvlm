#include <vector>
#include <string>
#include <functional> // std::function
#include <fstream>
#include <filesystem>

#include "tinycombination.hpp"
#include "tinyad.hpp"
#include "tinypbar.hpp"
#include "tinytimer.hpp"
#include "tinytest.hpp"
#include "npy.hpp"

#include "vlm.hpp"
#include "vlm_backend.hpp"
#include "vlm_types.hpp"
#include "vlm_kinematics.hpp"
#include "vlm_utils.hpp"

using namespace vlm;
using namespace linalg::ostream_overloads;
namespace fs = std::filesystem;

#define EXTERNAL_KINEMATICS 1

class HBVLM final: public Simulation {
    public:
        HBVLM(const std::string& backend_name, const Assembly& assembly, const i32 harmonics);
        ~HBVLM() = default;
        void run(f32 t_final, f32 omega, f32 vars_b);

        MultiTensor3fD colloc_d{backend->memory.get()};
        MultiTensor3fH colloc_h{backend->memory.get()};
        MultiTensor3fD normals_d{backend->memory.get()};
        MultiTensor2fD areas_d{backend->memory.get()};

        // Data (uharmonics = 2*m_harmonics + 1)
        Tensor2fD lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor2fD rhs{backend->memory.get()}; // (ns*nc) x uharmonics
        Tensor2fD gamma_coeffs{backend->memory.get()}; // (ns*nc) x uharmonics
        Tensor2fH gamma_coeffs_h{backend->memory.get()}; // (ns*nc) x uharmonics
        Tensor2fD gamma_wing{backend->memory.get()}; // (ns*nc) x time instances
        MultiTensor2fD gamma_wing_delta{backend->memory.get()}; // ns x nc
        Tensor2fD dgamma_wing_dt{backend->memory.get()}; // dgamma/dt

        Tensor2fD residual{backend->memory.get()}; // (ns*nc) x uharmonics
        Tensor2fH dft_h{backend->memory.get()}; // uharmonics x uharmonics
        Tensor2fD dft_d{backend->memory.get()};
        Tensor2fH ddft_h{backend->memory.get()}; // uharmonics x uharmonics
        Tensor2fD ddft_d{backend->memory.get()};
        std::vector<MultiTensor2fD> gamma_wake; // TODO: change this to something better
        MultiTensor3fD aero_forces{backend->memory.get()}; // ns*nc*3
        Tensor2fH coeffs_h{backend->memory.get()}; // 2 x coeffs
        Tensor2fD coeffs_d{backend->memory.get()}; // 2 x coeffs
        Tensor2fD coeffs_c_d{backend->memory.get()}; // 2 x coeffs
        MultiTensor3fD velocities{backend->memory.get()}; // ns*nc*3
        MultiTensor3fH velocities_h{backend->memory.get()}; // ns*nc*3

        MultiTensor2fH transforms_h{backend->memory.get()};
        MultiTensor2fD transforms{backend->memory.get()};
        
        std::unique_ptr<LU> solver;

        std::vector<i32> condition0;
        Assembly m_assembly;
        i32 m_harmonics;
        i32 m_coeffs;

    private:
        void alloc_buffers();
};

HBVLM::HBVLM(
    const std::string& backend_name,
    const Assembly& assembly,
    const i32 harmonics) : 
    Simulation(backend_name, assembly.mesh_filenames(), true),
    m_assembly(assembly),
    m_harmonics(harmonics),
    m_coeffs(2*harmonics+1)
{
    solver = backend->create_lu_solver();
    alloc_buffers();
}

// TODO: move this somewhere else
inline i64 total_panels(const MultiDim<2>& assembly_wing) {
    i64 total = 0;
    for (const auto& wing : assembly_wing) {
        total += wing[0] * wing[1];
    }
    return total;
}

void build_scaled_fourier_series(TensorView1fH& factors, f32 omega, f32 t) {
    TINY_ASSERT_EQ(factors.shape(0) % 2, 1);
    i64 harmonics = (factors.shape(0)-1) / 2;
    const f32 coeff_scaling0 = 1.f / std::sqrt(static_cast<f32>(factors.shape(0)));
    const f32 coeff_scaling = std::sqrt(2.f / static_cast<f32>(factors.shape(0)));
    factors(0) = coeff_scaling0;
    for (i64 i = 0; i < harmonics; i++) {
        const f32 k = (f32)(i+1);
        factors(2*i+1) = std::cos(omega * t * k) * coeff_scaling;
        factors(2*i+2) = std::sin(omega * t * k) * coeff_scaling;
    }
}

void HBVLM::alloc_buffers() {
    const i64 n = total_panels(assembly_wings);
    const i64 unknowns = 2 * m_harmonics + 1;
    for (i64 i = 0; i < unknowns; i++) {
        gamma_wake.emplace_back(backend->memory.get());
    }
    // Mesh
    MultiDim<3> panels_3D;
    MultiDim<2> panels_2D;
    MultiDim<3> verts_wing_3D;
    MultiDim<2> transforms_2D;
    for (const auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
        panels_2D.push_back({ns, nc});
        verts_wing_3D.push_back({ns+1, nc+1, 4});
        transforms_2D.push_back({4, 4});
    }
    normals_d.init(panels_3D);
    colloc_d.init(panels_3D);
    colloc_h.init(panels_3D);
    areas_d.init(panels_2D);

    // Data
    lhs.init({n, n});
    rhs.init({n, unknowns});
    gamma_coeffs.init({n, unknowns});
    gamma_coeffs_h.init({n, unknowns});
    gamma_wing.init({n, unknowns});
    gamma_wing_delta.init(panels_2D);
    dgamma_wing_dt.init({n, unknowns});
    residual.init({n, unknowns});
    dft_d.init({unknowns, unknowns});
    dft_h.init({unknowns, unknowns});
    ddft_d.init({unknowns, unknowns});
    ddft_h.init({unknowns, unknowns});
    coeffs_h.init({2, unknowns});
    coeffs_d.init({2, unknowns});
    coeffs_c_d.init({2, unknowns});
    velocities.init(panels_3D);
    velocities_h.init(panels_3D);
    aero_forces.init(panels_3D);
    transforms_h.init(transforms_2D);
    transforms.init(transforms_2D);
    solver->init(lhs.view());

    condition0.resize(assembly_wings.size()*assembly_wings.size());
}

void gamma_wake_from_coeffs(
    const TensorView2fD& gamma_wake,
    const TensorView2fD& gamma_coeffs,
    i32 harmonics,
    f32 tn,
    f32 omega,
    f32 dt,
    i64 iteration
)
{
    assert(gamma_coeffs.shape(0) == gamma_wake.shape(0));
    i64 wake_start = gamma_wake.shape(1) - iteration;
    for (i64 j = wake_start; j < gamma_wake.shape(1); j++) { // row
        for (i64 i = 0; i < gamma_wake.shape(0); i++) { // col
            f32 gamma_w = gamma_coeffs(i, 0);
            for (i64 h = 0; h < harmonics; h++) {
                const f32 omega_k = omega * (f32)(h+1);
                gamma_w += gamma_coeffs(i, 2*h+1) * std::cos(omega_k * (tn - (f32)(j - wake_start + 1)*dt));
                gamma_w += gamma_coeffs(i, 2*h+2) * std::sin(omega_k * (tn - (f32)(j - wake_start + 1)*dt));
            }
            gamma_wake(i, j) = gamma_w;
        }
    }
}

void anderson_acceleration(
    Backend* backend,
    const TensorView1fD& x0,
    const std::function<void(const TensorView1fD& x, const TensorView1fD& y)>& f,
    i32 max_iter = 100,
    f32 tol_res = 1e-6,
    i32 m = 3
)
{
    i64 n = x0.shape(0);
    Tensor2fD m_X_buf{backend->memory.get()};
    Tensor2fD m_G_buf{backend->memory.get()};
    Tensor2fD m_G_buf_k{backend->memory.get()}; // copy because lsq overwrites
    Tensor1fD m_x_curr{backend->memory.get()};
    Tensor1fD m_x_new{backend->memory.get()};
    Tensor1fD m_g_curr{backend->memory.get()};
    Tensor1fD m_g_new{backend->memory.get()};
    Tensor1fD m_gamma{backend->memory.get()};
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

    std::printf("Anderson fixed point converged in %d iterations\n", k);
}

void HBVLM::run(f32 t_start, f32 omega, f32 vars_b) {
    const tiny::ScopedTimer timer("HBVLM::run");
    const f32 period = 2.0f * PI_f / omega;
    const f32 period_interval = period / (f32)m_coeffs;
    const f32 rho = 1.0f; // TODO: take this as input

    for (const auto& [vwing, vwing_init_d, vwing_init_h] : zip(verts_wing.views(), verts_wing_init.views(), verts_wing_init_h.views())) {
        // Nondimensionalize the coordinates
        backend->blas->scal(1.0f / vars_b, vwing_init_d.slice(All, All, Range{0, 3}).reshape(3*vwing_init_d.shape(0)*vwing_init_d.shape(1)));
        vwing_init_d.to(vwing_init_h);
        vwing_init_d.to(vwing);
    }
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);

    // Compute the fixed time step
    const auto& verts_first_wing = verts_wing_init_h.views()[0];
    const f32 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    const f32 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 1);
    const f32 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 2);
    const f32 last_panel_chord = std::sqrt(dx*dx + dy*dy + dz*dz);
    // const f32 dt = last_panel_chord; // ideal chord
    TINY_ASSERT_GT(period_interval, last_panel_chord); // TODO: check what happens if this is not the case
    const f32 dt = period_interval / std::ceil(period_interval / last_panel_chord); // period interval must be a multiple of dt
    std::printf("period_interval: %f | chord: %f | dt: %f\n", period_interval, last_panel_chord, dt);

    const i64 max_t_steps = static_cast<i64>((t_start+period+dt) / dt);

    // Allocate the wake geometry
    {
        MultiDim<2> wake_panels_2D;
        MultiDim<3> verts_wake_3D;
        for (const auto& [ns, nc] : assembly_wings) {
            wake_panels_2D.push_back({ns, max_t_steps-1});
            verts_wake_3D.push_back({ns+1, max_t_steps, 4});
        }
        verts_wake.init(verts_wake_3D);
        for (auto& gw : gamma_wake) gw.init(wake_panels_2D);
    }

    // Precompute constant lhs since we have a rigid wing
    lhs.view().fill(0.f);
    backend->lhs_assemble(lhs.view(), colloc_d.views(), normals_d.views(), verts_wing.views(), verts_wake.views(), condition0 , 0);
    solver->factorize(lhs.view());

    // Pregenerate wake
    for (i32 i = 0; i < max_t_steps; i++) {
        const f32 t = (f32)i * dt;

        // parallel for
        for (i64 m = 0; m < assembly_wings.size(); m++) {
            auto transform = m_assembly.surface_kinematics()[m]->transform(t);
            transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
            transforms_h.views()[m].to(transforms.views()[m]);
        }

        backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
        backend->wake_shed(verts_wing.views(), verts_wake.views(), i);
 
        if (i == 0) { // todo: check if this is necessary for the velocity computation
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
            for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);
        }
    }

    // Assemble orthogonal DFT matrix (TODO: move to a standalone function)
    {
        const i32 unknowns = 2 * m_harmonics + 1;
        const f32 sqrt_unknowns = 1.f / std::sqrt(static_cast<f32>(unknowns));
        const f32 sqrt_unknowns_2 = std::sqrt(2.f / static_cast<f32>(unknowns));
        auto& dft_hv = dft_h.view();
        auto& dft_dv = dft_d.view();
        auto ddft_hv = ddft_h.view();
        auto ddft_dv = ddft_d.view();

        for (i64 i = 0; i < unknowns; i++) {
            dft_hv(i, 0) = sqrt_unknowns;
            ddft_hv(i, 0) = 0.0f;
        }
        for (i64 j = 1; j < unknowns; j += 2) {
            const f32 k = ((f32)j + 1.f) / 2.f;
            for (i64 i = 0; i < unknowns; i++) {
                const f32 tn = ((f32)i / (f32)unknowns) * period;
                dft_hv(i, j) = std::cos(omega * tn * k) * sqrt_unknowns_2;
                dft_hv(i, j+1) = std::sin(omega * tn * k) * sqrt_unknowns_2;
                ddft_hv(i, j) = - omega * k * std::sin(omega * tn * k) * sqrt_unknowns_2;
                ddft_hv(i, j+1) = omega * k * std::cos(omega * tn * k) * sqrt_unknowns_2;
            }
        }
        dft_hv.to(dft_dv);
        ddft_hv.to(ddft_dv);
    }

    auto hb_vlm_iter = [&](
        const TensorView1fD& _gamma_in, 
        const TensorView1fD& _gamma_out
    ){
        auto gamma_in  = _gamma_in.reshape(lhs.view().shape(0), m_coeffs);
        auto gamma_out = _gamma_out.reshape(lhs.view().shape(0), m_coeffs);

        rhs.view().fill(0.f);
        // Note this only work on a single wing
        auto te_gamma = gamma_in
            .reshape(colloc_d.views()[0].shape(0), colloc_d.views()[0].shape(1), gamma_in.shape(1))
            .slice(All, -1, All);
            
        // For each unknown we fill their respective rhs column in a matrix free fashion
        for (i32 s = 0; s < m_coeffs; s++) {
            const f32 t_final = t_start + period * (f32)s / (f32)(m_coeffs);
            const i64 t_steps = static_cast<i64>(std::round(t_final / dt));
            const f32 t = (f32)t_steps * dt; // t_final rounded to nearest dt

            for (i64 m = 0; m < assembly_wings.size(); m++) {
                auto transform = m_assembly.surface_kinematics()[m]->transform(t);
                transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
                transforms_h.views()[m].to(transforms.views()[m]);
            }

            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());

            // parallel for
            for (i64 m = 0; m < assembly_wings.size(); m++) {
                const auto transform_node = m_assembly.surface_kinematics()[m];
                const auto& colloc_h_m = colloc_h.views()[m];
                const auto& velocities_h_m = velocities_h.views()[m];
                const auto& velocities_m = velocities.views()[m];
                const auto mat_transform_dual = transform_node->transform_dual(t);

                for (i64 j = 0; j < colloc_h_m.shape(1); j++) {
                    for (i64 i = 0; i < colloc_h_m.shape(0); i++) {
                        auto local_velocity = -transform_node->linear_velocity(mat_transform_dual, {colloc_h_m(i, j, 0), colloc_h_m(i, j, 1), colloc_h_m(i, j, 2)});
                        velocities_h_m(i, j, 0) = local_velocity.x;
                        velocities_h_m(i, j, 1) = local_velocity.y;
                        velocities_h_m(i, j, 2) = local_velocity.z;
                    }
                }
                velocities_h_m.to(velocities_m);
            }

            auto rhs_s = rhs.view().slice(All, s);
            backend->rhs_assemble_velocities(rhs_s, normals_d.views(), velocities.views());
            backend->gamma_wake_from_coeffs(gamma_wake[s].views()[0], te_gamma, m_harmonics, t, omega, dt, t_steps);
            backend->rhs_assemble_wake_influence(rhs_s, gamma_wake[s].views(), colloc_d.views(), normals_d.views(), verts_wake.views(), m_assembly.lifting(), (i32)t_steps);
        } // unknowns for loop

        // Apply the the dft matrix to the rhs
        // A @ G @ D^T = R <=> A @ G = R @ D (since D^-1 = D^T)
        backend->blas->gemm(1.0f, rhs.view(), dft_d.view(), 0.0f, gamma_out);
        // Solve to obtain the fourier coefficients for each panel
        solver->solve(lhs.view(), gamma_out);
    };
    
    auto& gamma_coeffs_v = gamma_coeffs.view();
    gamma_coeffs_v.fill(0.f);

    anderson_acceleration(
        backend.get(),
        gamma_coeffs_v.reshape(gamma_coeffs_v.shape(0)*gamma_coeffs_v.shape(1)),
        hb_vlm_iter,
        100, // max iterations
        1e-6, // tolerance
        5 // history
    );
    
    // Compute the forces coefficients
    {
        // Maybe we should use mesh dims throughout the code ...
        i64 c_ns = colloc_d.views()[0].shape(0);
        i64 c_nc = colloc_d.views()[0].shape(1);
        backend->blas->gemm(1.0f, gamma_coeffs.view(), dft_d.view(), 0.0f, gamma_wing.view(), BLAS::Trans::No, BLAS::Trans::Yes);
        backend->blas->gemm(1.0f, gamma_coeffs.view(), ddft_d.view(), 0.0f, dgamma_wing_dt.view(), BLAS::Trans::No, BLAS::Trans::Yes);
        for (i32 s = 0; s < m_coeffs; s++) {
            const f32 t_final = period * (f32)s / (f32)(m_coeffs);
            const i64 t_steps = static_cast<i64>(std::round(t_final / dt));
            const f32 t = (f32)t_steps * dt; // t_final rounded to nearest dt

            for (i64 m = 0; m < assembly_wings.size(); m++) {
                auto transform = m_assembly.surface_kinematics()[m]->transform(t);
                transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
                transforms_h.views()[m].to(transforms.views()[m]);
            }

            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());

            // TODO: move this into its own standalone function
            for (i64 m = 0; m < assembly_wings.size(); m++) {
                const auto transform_node = m_assembly.surface_kinematics()[m];
                const auto& colloc_h_m = colloc_h.views()[m];
                const auto& velocities_h_m = velocities_h.views()[m];
                const auto& velocities_m = velocities.views()[m];
                const auto mat_transform_dual = transform_node->transform_dual(t);

                for (i64 j = 0; j < colloc_h_m.shape(1); j++) {
                    for (i64 i = 0; i < colloc_h_m.shape(0); i++) {
                        auto local_velocity = -transform_node->linear_velocity(mat_transform_dual, {colloc_h_m(i, j, 0), colloc_h_m(i, j, 1), colloc_h_m(i, j, 2)});
                        velocities_h_m(i, j, 0) = local_velocity.x;
                        velocities_h_m(i, j, 1) = local_velocity.y;
                        velocities_h_m(i, j, 2) = local_velocity.z;
                    }
                }
                velocities_h_m.to(velocities_m);
            }
            const linalg::float3 freestream = -m_assembly.kinematics()->linear_velocity(t, {0.f, 0.f, 0.f});

            auto gamma_wing_s = gamma_wing.view().slice(All, s).reshape(c_ns, c_nc);
            auto dgamma_wing_dt_s = dgamma_wing_dt.view().slice(All, s).reshape(c_ns, c_nc);
            gamma_wing_s.to(gamma_wing_delta.views()[0]);
            backend->blas->axpy(
                -1.0f,
                gamma_wing_s.slice(All, Range{0, -2}),
                gamma_wing_delta.views()[0].slice(All, Range{1, -1})
            );
            backend->forces_unsteady2(
                verts_wing.views()[0],
                gamma_wing_delta.views()[0],
                dgamma_wing_dt_s,
                velocities.views()[0],
                areas_d.views()[0],
                normals_d.views()[0],
                aero_forces.views()[0]
            ); 
            coeffs_h.view()(0, s) = backend->coeff_cl_multibody(
                aero_forces.views(),
                areas_d.views(),
                freestream,
                rho
            );
            const auto transform_node0 = m_assembly.surface_kinematics()[0];
            const auto transform_mat = transform_node0->transform(t);
            auto ref_pt = linalg::mul(transform_mat, {0.5f, 0.0f, 0.0f, 1.0f});
            coeffs_h.view()(1, s) = backend->coeff_cm_multibody(
                aero_forces.views(),
                verts_wing.views(),
                areas_d.views(),
                {ref_pt.x, ref_pt.y, ref_pt.z},
                freestream,
                rho
            ).y;
        }
        coeffs_h.view().to(coeffs_d.view());
        {
            npy::npy_data_ptr<f32> bin_tdata;
            bin_tdata.data_ptr = coeffs_h.ptr();
            bin_tdata.shape = {(npy::ndarray_len_t)coeffs_h.view().shape(0), (npy::ndarray_len_t)coeffs_h.view().shape(1)};
            bin_tdata.fortran_order = true;
            npy::write_npy("hbvlm_t.npy", bin_tdata);
        }
    }

    #ifndef EXTERNAL_KINEMATICS
    {
        backend->blas->gemm(1.0f, coeffs_d.view(), dft_d.view(), 0.0f, coeffs_c_d.view());
        coeffs_c_d.view().to(coeffs_h.view());

        npy::npy_data_ptr<f32> bin_data;
        bin_data.data_ptr = coeffs_h.ptr();
        bin_data.shape = {(npy::ndarray_len_t)coeffs_h.view().shape(0), (npy::ndarray_len_t)coeffs_h.view().shape(1)};
        bin_data.fortran_order = true;
        npy::write_npy("hbvlm.npy", bin_data);
    }

    // Compute time domain solution from the obtained fourier coeffs
    {
        auto& gamma_coeffs_hv = gamma_coeffs_h.view();
        auto coeffs_cl_v = coeffs_h.view().slice(0, All);
        auto coeffs_cm_v = coeffs_h.view().slice(1, All);
        gamma_coeffs.view().to(gamma_coeffs_hv);
        std::ofstream hb_data("hbvlm_data_" + backend->name + ".txt");

        const i32 unknowns = 2 * m_harmonics + 1;
        const f32 sqrt_unknowns = 1.f / std::sqrt(static_cast<f32>(unknowns));
        const f32 sqrt_unknowns_2 = std::sqrt(2.f / static_cast<f32>(unknowns));

        for (i64 i = 0; i < max_t_steps; i++) {
            const f32 t = (f32)i * dt;
            f32 gamma = gamma_coeffs_hv(0,0) * sqrt_unknowns;
            f32 cl = coeffs_cl_v(0) * sqrt_unknowns;
            f32 cm = coeffs_cm_v(0) * sqrt_unknowns;
            f32 dgamma = 0.f;
            for (i64 j = 0; j < m_harmonics; j++) {
                const f32 k = (f32)(j+1);
                const f32 c = std::cos(omega * t * k) * sqrt_unknowns_2;
                const f32 s = std::sin(omega * t * k) * sqrt_unknowns_2;
                gamma += gamma_coeffs_hv(0, 2*j+1) * c;
                gamma += gamma_coeffs_hv(0, 2*j+2) * s;
                dgamma += - gamma_coeffs_hv(0, 2*j+1) * omega * k * s;
                dgamma += gamma_coeffs_hv(0, 2*j+2) * omega * k * c;
                cl += coeffs_cl_v(2*j+1) * c;
                cl += coeffs_cl_v(2*j+2) * s;
                cm += coeffs_cm_v(2*j+1) * c;
                cm += coeffs_cm_v(2*j+2) * s;
            }
            hb_data << t << " " << gamma << " " << cl << " " << cm << " " << dgamma << "\n";
        }
    }
    #endif
}

int main(int argc, char** argv) {
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_10x5.x"};
    // const std::vector<std::string> backends = get_available_backends();
    const std::vector<std::string> backends = {"cpu"};

    auto solvers = tiny::make_combination(meshes, backends);
 
    // Geometry
    const f32 b = 0.5f; // half chord

    #ifndef EXTERNAL_KINEMATICS
    // Define simulation length
    const f32 cycles = 9.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    // const f32 k = 0.25f; // reduced frequency
    // const f32 omega = k * u_inf / b;
    const f32 omega = 0.145;
    const i32 H = 15;
    const f32 t_final = cycles * 2.0f * PI_f / omega;
    // Periodic pitching
    // const f32 amplitude = to_radians(3.f); // amplitude in radians
    const f32 amplitude = 0.05f;
    KinematicsTree kinematics_tree;
    auto fs = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    });
    auto heave = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({0.f, 0.f, 0.0f * t});
    })->after(fs);
    auto pitch = kinematics_tree.add([=](const fwd::Float& t) {
        fwd::Float alpha = amplitude * fwd::sin(omega * t);
        return rotation_matrix<fwd::Float>(
            {0.5f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            alpha
        );
    })->after(heave);
    #else
    if (argc != 3) { // hbvlm <omega> <harmonics>
        return 1;
    }
    const f32 cycles = 9.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 omega = std::stof(argv[1]);
    const i32 H = std::stoi(argv[2]);
    const f32 t_final = cycles * 2.0f * PI_f / omega;
    const f32 a_h = -0.5f;

    std::printf("t_final: %f, omega: %.8f, harmonics: %i\n", t_final, omega, H);
    
    fs::path npy_path = fs::path(R"(C:\Users\samay\Documents\GitHub\tinyvlm\build\windows\x64\release\kin_coeffs.npy)");
    npy::npy_data kin_coeffs_npy = npy::read_npy<f64>(npy_path.string());
    TINY_ASSERT_EQ(kin_coeffs_npy.fortran_order, true);
    TINY_ASSERT_EQ(kin_coeffs_npy.shape[0], (npy::ndarray_len_t)(2));
    TINY_ASSERT_EQ(kin_coeffs_npy.shape[1], (npy::ndarray_len_t)(2*H+1));

    f64* h_ptr = kin_coeffs_npy.data.data();
    f64* alpha_ptr = h_ptr + 1;
    i32 ld = 2;

    KinematicsTree kinematics_tree;
    auto fs = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    });
    auto heave = kinematics_tree.add([=](const fwd::Float& t) {
        fwd::Float z = (f32)h_ptr[0];
        for (i32 h = 0; h < H; h++) {
            f32 k = (f32)(h+1);
            z += fwd::cos(omega * t * k) * (f32)h_ptr[(2*h+1)*ld];
            z += fwd::sin(omega * t * k) * (f32)h_ptr[(2*h+2)*ld];
        }
        return translation_matrix<fwd::Float>({0.f, 0.f, -z});
    })->after(fs);
    auto pitch = kinematics_tree.add([=](const fwd::Float& t) {
        fwd::Float alpha = (f32)alpha_ptr[0];
        for (i32 h = 0; h < H; h++) {
            f32 k = (f32)(h+1);
            alpha += fwd::cos(omega * t * k) * (f32)alpha_ptr[(2*h+1)*ld];
            alpha += fwd::sin(omega * t * k) * (f32)alpha_ptr[(2*h+2)*ld];
        }

        return rotation_matrix<fwd::Float>(
            {1.f + a_h, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}, 
            alpha);
    })->after(heave);
    #endif
 
    for (const auto& [mesh_name, backend_name] : solvers) {
        Assembly assembly(fs);
        assembly.add(mesh_name, pitch);
        HBVLM simulation{backend_name, assembly, H};
        simulation.run(t_final, omega, b);
    }
    return 0;
}
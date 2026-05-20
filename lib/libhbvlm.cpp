#include <vector>
#include <string>
#include <functional> // std::function
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
#include "vlm_io.hpp"
#include "vlm_solvers.hpp"

using namespace vlm;
using namespace linalg::ostream_overloads;
namespace py = pybind11;

// Delete this ASAP
#define TRY_CATCH_VOID_CALL(expr) \
    try { \
        expr; \
    } catch (const std::exception& ex) { \
        std::cerr << "Exception in " #expr ": " << ex.what() << std::endl; \
        std::exit(1); \
    } catch (...) { \
        std::cerr << "Unknown exception in " #expr << std::endl; \
        std::exit(1); \
    }

// TODO: move this somewhere else
inline i64 total_panels(const MultiDim<2>& assembly_wing) {
    i64 total = 0;
    for (const auto& wing : assembly_wing) {
        total += wing[0] * wing[1];
    }
    return total;
}

void build_scaled_fourier_series(TensorView1dH& factors, f64 omega, f64 t) {
    TINY_ASSERT_EQ(factors.shape(0) % 2, 1);
    i64 harmonics = (factors.shape(0)-1) / 2;
    const f64 coeff_scaling0 = 1.f / std::sqrt(static_cast<f64>(factors.shape(0)));
    const f64 coeff_scaling = std::sqrt(2.f / static_cast<f64>(factors.shape(0)));
    factors(0) = coeff_scaling0;
    for (i64 i = 0; i < harmonics; i++) {
        const f64 k = (f64)(i+1);
        factors(2*i+1) = std::cos(omega * t * k) * coeff_scaling;
        factors(2*i+2) = std::sin(omega * t * k) * coeff_scaling;
    }
}

class HBVLM {
    public:
        HBVLM(const std::string& backend_name, const std::string& filename);
        ~HBVLM() = default;
        void init(i32 harmonics, f64 scaling);
        void run(f64 t_final, f64 omega, f64 vars_b, const Assembly<f64>& assembly, py::array_t<double>& forces_t);

        std::unique_ptr<Backend> backend;
        MultiDim<2> assembly_wings;

        MultiTensor3dD verts_wing_init{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3dH verts_wing_init_h{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3dD verts_wing{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3dH verts_wing_h{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3dD verts_wake{backend->memory.get()}; // (nw+1)*(ns+1)*3
        MultiTensor3dH verts_wake_h{backend->memory.get()}; // (nw+1)*(ns+1)*3

        MultiTensor3dD colloc_d{backend->memory.get()};
        MultiTensor3dH colloc_h{backend->memory.get()};
        MultiTensor3dD normals_d{backend->memory.get()};
        MultiTensor2dD areas_d{backend->memory.get()};

        // Data (uharmonics = 2*m_harmonics + 1)
        Tensor2dD lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor2dD rhs{backend->memory.get()}; // (ns*nc) x uharmonics
        Tensor2dD gamma_coeffs{backend->memory.get()}; // (ns*nc) x uharmonics
        Tensor2dD gamma_wing{backend->memory.get()}; // (ns*nc) x time instances
        MultiTensor2dD gamma_wing_delta{backend->memory.get()}; // ns x nc
        Tensor2dD dgamma_wing_dt{backend->memory.get()}; // dgamma/dt

        Tensor2dD residual{backend->memory.get()}; // (ns*nc) x uharmonics
        Tensor2dH dft_h{backend->memory.get()}; // uharmonics x uharmonics
        Tensor2dD dft_d{backend->memory.get()};
        Tensor2dH ddft_h{backend->memory.get()}; // uharmonics x uharmonics
        Tensor2dD ddft_d{backend->memory.get()};
        std::vector<MultiTensor2dD> gamma_wake; // TODO: change this to something better
        MultiTensor3dD aero_forces{backend->memory.get()}; // ns*nc*3
        MultiTensor3dD velocities{backend->memory.get()}; // ns*nc*3
        MultiTensor3dH velocities_h{backend->memory.get()}; // ns*nc*3

        MultiTensor2dH transforms_h{backend->memory.get()};
        MultiTensor2dD transforms{backend->memory.get()};
        
        std::unique_ptr<LU> solver;

        std::vector<i32> condition0;
        i32 m_harmonics;
        i32 m_coeffs;

    private:
        void alloc_buffers();
};

HBVLM::HBVLM(const std::string& backend_name, const std::string& filename) : backend(create_backend(backend_name)) {
    const std::vector<std::string> meshes = {filename};
    // Read the sizes of all the meshes
    for (const auto& m_name : meshes) {
        const MeshIO mesh_io{"plot3d"}; // TODO, infer from mesh_name
        auto [nc,ns] = mesh_io.get_dims(m_name);
        assembly_wings.push_back({ns, nc});
    }

    MultiDim<3> verts_wing_3D;
    for (const auto& [ns, nc] : assembly_wings) {
        verts_wing_3D.push_back({ns+1, nc+1, 4});
    }

    verts_wing_init.init(verts_wing_3D);
    verts_wing_init_h.init(verts_wing_3D);
    verts_wing.init(verts_wing_3D);
    verts_wing_h.init(verts_wing_3D);

    // Read the files
    for (i64 i = 0; i < meshes.size(); i++) {
        const MeshIO mesh_io{"plot3d"};
        mesh_io.read(meshes[i], verts_wing_init_h.views()[i], true);
    }

    for (const auto& [init_h, init_d] : zip(verts_wing_init_h.views(), verts_wing_init.views())) {
        init_h.slice(All, All, 3).fill(1.f);
        init_h.to(init_d);
    }

    solver = backend->create_lu_solver();
}

void HBVLM::init(i32 harmonics, f64 scaling) {
    m_harmonics = harmonics;
    m_coeffs = 2*harmonics+1;
    alloc_buffers();

    for (const auto& [vwing_init_h, vwing_init_d] : zip(verts_wing_init_h.views(), verts_wing_init.views())) {
        // Nondimensionalize the coordinates
        backend->blas->scal(scaling, vwing_init_d.slice(All, All, Range{0, 3}).reshape(3*vwing_init_d.shape(0)*vwing_init_d.shape(1)));
        vwing_init_d.to(vwing_init_h);
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
    gamma_coeffs.view().fill(0.f);
    gamma_wing.init({n, unknowns});
    gamma_wing_delta.init(panels_2D);
    dgamma_wing_dt.init({n, unknowns});
    residual.init({n, unknowns});
    dft_d.init({unknowns, unknowns});
    dft_h.init({unknowns, unknowns});
    ddft_d.init({unknowns, unknowns});
    ddft_h.init({unknowns, unknowns});
    velocities.init(panels_3D);
    velocities_h.init(panels_3D);
    aero_forces.init(panels_3D);
    transforms_h.init(transforms_2D);
    transforms.init(transforms_2D);
    solver->init(lhs.view());

    condition0.resize(assembly_wings.size()*assembly_wings.size());
}

// This wont work on GPU
void gamma_wake_from_coeffs(
    const TensorView2dD& gamma_wake,
    const TensorView2dD& gamma_coeffs,
    i32 harmonics,
    f64 tn,
    f64 omega,
    f64 dt,
    i64 iteration
)
{
    assert(gamma_coeffs.shape(0) == gamma_wake.shape(0));
    i64 wake_start = gamma_wake.shape(1) - iteration;
    for (i64 j = wake_start; j < gamma_wake.shape(1); j++) { // row
        for (i64 i = 0; i < gamma_wake.shape(0); i++) { // col
            f64 gamma_w = gamma_coeffs(i, 0);
            for (i64 h = 0; h < harmonics; h++) {
                const f64 omega_k = omega * (f64)(h+1);
                gamma_w += gamma_coeffs(i, 2*h+1) * std::cos(omega_k * (tn - (f64)(j - wake_start + 1)*dt));
                gamma_w += gamma_coeffs(i, 2*h+2) * std::sin(omega_k * (tn - (f64)(j - wake_start + 1)*dt));
            }
            gamma_wake(i, j) = gamma_w;
        }
    }
}

void HBVLM::run(f64 t_start, f64 omega, f64 vars_b, const Assembly<f64>& assembly, py::array_t<double>& forces_t) {
    // const tiny::ScopedTimer timer("HBVLM::run");

    const f64 period = 2.0f * PI_f / omega;
    const f64 period_interval = period / (f64)m_coeffs;
    const f64 rho = 1.0f; // TODO: take this as input

    for (const auto& [vwing_d, vwing_h, vwing_init_d] : zip(verts_wing.views(), verts_wing_h.views(), verts_wing_init.views())) {
        vwing_init_d.to(vwing_d);
        vwing_d.to(vwing_h);
    }
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());
    for (const auto& [c_h, c_d] : zip(colloc_h.views(), colloc_d.views())) c_d.to(c_h);

    // Compute the fixed time step
    const auto& verts_first_wing = verts_wing_h.views()[0];
    const f64 dx = verts_first_wing(0, -1, 0) - verts_first_wing(0, -2, 0);
    const f64 dy = verts_first_wing(0, -1, 1) - verts_first_wing(0, -2, 1);
    const f64 dz = verts_first_wing(0, -1, 2) - verts_first_wing(0, -2, 2);
    const f64 last_panel_chord = std::sqrt(dx*dx + dy*dy + dz*dz);
    // const f64 dt = last_panel_chord; // ideal chord
    TINY_ASSERT_GT(period_interval, last_panel_chord); // TODO: check what happens if this is not the case
    const f64 dt = period_interval / std::ceil(period_interval / last_panel_chord); // period interval must be a multiple of dt
    // std::printf("period_interval: %f | chord: %f | dt: %f\n", period_interval, last_panel_chord, dt);

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
        const f64 t = (f64)i * dt;

        // parallel for
        for (i64 m = 0; m < assembly_wings.size(); m++) {
            auto transform = assembly.surface_kinematics()[m]->transform(t);
            transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
            transforms_h.views()[m].to(transforms.views()[m]);
        }

        backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
        backend->wake_shed(verts_wing.views(), verts_wake.views(), i);
    }

    // Assemble orthogonal DFT matrix (TODO: move to a standalone function)
    {
        const i32 unknowns = 2 * m_harmonics + 1;
        const f64 sqrt_unknowns = 1.f / std::sqrt(static_cast<f64>(unknowns));
        const f64 sqrt_unknowns_2 = std::sqrt(2.f / static_cast<f64>(unknowns));
        auto& dft_hv = dft_h.view();
        auto& dft_dv = dft_d.view();
        auto ddft_hv = ddft_h.view();
        auto ddft_dv = ddft_d.view();

        for (i64 i = 0; i < unknowns; i++) {
            dft_hv(i, 0) = sqrt_unknowns;
            ddft_hv(i, 0) = 0.0f;
        }
        for (i64 j = 1; j < unknowns; j += 2) {
            const f64 k = ((f64)j + 1.f) / 2.f;
            for (i64 i = 0; i < unknowns; i++) {
                const f64 tn = ((f64)i / (f64)unknowns) * period;
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
        const TensorView1dD& _gamma_in, 
        const TensorView1dD& _gamma_out
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
            const f64 t_final = t_start + period * (f64)s / (f64)(m_coeffs);
            const i64 t_steps = static_cast<i64>(std::round(t_final / dt));
            const f64 t = (f64)t_steps * dt; // t_final rounded to nearest dt

            for (i64 m = 0; m < assembly_wings.size(); m++) {
                auto transform = assembly.surface_kinematics()[m]->transform(t);
                transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
                transforms_h.views()[m].to(transforms.views()[m]);
            }

            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());

            // parallel for
            for (i64 m = 0; m < assembly_wings.size(); m++) {
                const auto transform_node = assembly.surface_kinematics()[m];
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
            backend->rhs_assemble_wake_influence(rhs_s, gamma_wake[s].views(), colloc_d.views(), normals_d.views(), verts_wake.views(), assembly.lifting(), (i32)t_steps);
        } // unknowns for loop

        // Apply the the dft matrix to the rhs
        // A @ G @ D^T = R <=> A @ G = R @ D (since D^-1 = D^T)
        backend->blas->gemm(1.0f, rhs.view(), dft_d.view(), 0.0f, gamma_out);
        // Solve to obtain the fourier coefficients for each panel
        solver->solve(lhs.view(), gamma_out);
    };
    
    auto& gamma_coeffs_v = gamma_coeffs.view();
    // gamma_coeffs_v.fill(0.f);

    anderson_acceleration(
        backend.get(),
        gamma_coeffs_v.reshape(gamma_coeffs_v.shape(0)*gamma_coeffs_v.shape(1)),
        hb_vlm_iter,
        100, // max iterations
        1e-9, // tolerance
        10 // history
    );
    
    auto forces_t_v = forces_t.mutable_unchecked<2>();

    // Compute the forces coefficients
    {
        // Maybe we should use mesh dims throughout the code ...
        i64 c_ns = colloc_d.views()[0].shape(0);
        i64 c_nc = colloc_d.views()[0].shape(1);
        backend->blas->gemm(1.0f, gamma_coeffs.view(), dft_d.view(), 0.0f, gamma_wing.view(), BLAS::Trans::No, BLAS::Trans::Yes);
        backend->blas->gemm(1.0f, gamma_coeffs.view(), ddft_d.view(), 0.0f, dgamma_wing_dt.view(), BLAS::Trans::No, BLAS::Trans::Yes);
        for (i32 s = 0; s < m_coeffs; s++) {
            const f64 t_final = period * (f64)s / (f64)(m_coeffs);
            const i64 t_steps = static_cast<i64>(std::round(t_final / dt));
            const f64 t = (f64)t_steps * dt; // t_final rounded to nearest dt

            for (i64 m = 0; m < assembly_wings.size(); m++) {
                auto transform = assembly.surface_kinematics()[m]->transform(t);
                transform.store(transforms_h.views()[m].ptr(), transforms_h.views()[m].stride(1));
                transforms_h.views()[m].to(transforms.views()[m]);
            }

            backend->displace_wing(transforms.views(), verts_wing.views(), verts_wing_init.views());
            backend->mesh_metrics(0.0f, verts_wing.views(), colloc_d.views(), normals_d.views(), areas_d.views());

            // TODO: move this into its own standalone function
            for (i64 m = 0; m < assembly_wings.size(); m++) {
                const auto transform_node = assembly.surface_kinematics()[m];
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
            const linalg::double3 freestream = -assembly.kinematics()->linear_velocity(t, {0.f, 0.f, 0.f});

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
            forces_t_v(0, s) = backend->coeff_cl_multibody(
                aero_forces.views(),
                areas_d.views(),
                freestream,
                rho
            );
            const auto transform_node0 = assembly.surface_kinematics()[0];
            const auto transform_mat = transform_node0->transform(t);
            auto ref_pt = linalg::mul(transform_mat, {0.5f, 0.0f, 0.0f, 1.0f});
            forces_t_v(1, s) = backend->coeff_cm_multibody(
                aero_forces.views(),
                verts_wing.views(),
                areas_d.views(),
                {ref_pt.x, ref_pt.y, ref_pt.z},
                freestream,
                rho
            ).y;
        }
    }
}

PYBIND11_MODULE(libhbvlm, m) {
    py::class_<HBVLM>(m, "HBVLM")
        .def(py::init<const std::string&, const std::string&>())
        .def("init", &HBVLM::init)
        .def("run", [](HBVLM& self, f64 omega, py::array_t<double> dyn_f, py::array_t<double> force_t) {
            const f64 b = 0.5f;
            const f64 cycles = 3.0f;
            const f64 u_inf = 1.0f;
            const f64 t_final = cycles * 2.0f * PI_f / omega;
            const f64 a_h = -0.5f;
            const i32 H = self.m_harmonics;

            auto r = dyn_f.unchecked<2>();

            KinematicsTree<f64> kinematics_tree;
            auto fs = kinematics_tree.add([=](const fwd::Double& t) {
                return translation_matrix<fwd::Double>({-u_inf * t, 0.f, 0.f});
            });
            
            auto heave = kinematics_tree.add([=](const fwd::Double& t) {
                fwd::Double z = r(0, 0);
                for (i32 h = 0; h < H; h++) {
                    f64 k = (f64)(h+1);
                    z += fwd::cos(omega * t * k) * r(0, 2*h+1);
                    z += fwd::sin(omega * t * k) * r(0, 2*h+2);
                }
                return translation_matrix<fwd::Double>({0.f, 0.f, -z});
            })->after(fs);
            
            auto pitch = kinematics_tree.add([=](const fwd::Double& t) {
                fwd::Double alpha = r(1, 0);
                for (i32 h = 0; h < H; h++) {
                    f64 k = (f64)(h+1);
                    alpha += fwd::cos(omega * t * k) * r(1, 2*h+1);
                    alpha += fwd::sin(omega * t * k) * r(1, 2*h+2);
                }
                return rotation_matrix<fwd::Double>(
                    {1.f + a_h, 0.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f}, 
                    alpha);
            })->after(heave);

            Assembly<f64> assembly(fs);
            assembly.add("your_mesh_name_here.x", pitch);  // Pass actual mesh name
            self.run(t_final, omega, b, assembly, force_t);
        });
}
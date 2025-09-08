#include "vlm_backend_cpu.hpp"
#include "vlm_backend_cpu_kernels_ispc.h"

#include "tinytimer.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_executor.hpp" // includes taskflow/taskflow.hpp

#include <algorithm> // std::fill

#include <taskflow/algorithm/for_each.hpp>

using namespace vlm;
using namespace linalg::ostream_overloads;

/// @brief Assemble the left hand side matrix
/// @details
/// Assemble the left hand side matrix of the VLM system. The matrix is
/// assembled in column major order. The matrix is assembled for each lifting
/// surface of the system
/// @param lhs left hand side matrix
/// @param colloc collocation points for all surfaces
/// @param normals normals of all surfaces
/// @param verts_wing vertices of the wing surfaces
/// @param verts_wake vertices of the wake surfaces
/// @param iteration iteration number (VLM = 1, UVLM = [0 ... N tsteps])
void BackendCPU::lhs_assemble(TensorView2dD& lhs, const MultiTensorView3dD& colloc, const MultiTensorView3dD& normals, const MultiTensorView3dD& verts_wing, const MultiTensorView3dD& verts_wake, std::vector<i32>& condition, i32 iteration) {
    // tiny::ScopedTimer timer("LHS");
    std::fill(condition.begin(), condition.end(), 0); // reset conditon increment vars

    tf::Taskflow graph;

    auto begin = graph.placeholder();
    auto end = graph.placeholder();

    i64 offset_j = 0;
    for (i32 m_j = 0; m_j < colloc.size(); m_j++) {
        i64 offset_i = 0;
        for (i32 m_i = 0; m_i < colloc.size(); m_i++) {
            const i64 condition_idx = m_i + m_j * colloc.size();
            const auto& colloc_i = colloc[m_i];
            const auto& colloc_j = colloc[m_j];
            const auto& normals_i = normals[m_i];
            const auto& verts_wing_j = verts_wing[m_j];
            const auto& verts_wake_j = verts_wake[m_j];

            f64* lhs_section = lhs.ptr() + offset_i + offset_j * lhs.stride(1);
            
            const i64 zero = 0;
            const i64 end_wing = (colloc_j.shape(1) - 1) * colloc_j.shape(0);
            
            auto wing_pass = graph.for_each_index(zero, end_wing, [=] (i64 lidx) {
                f64* lhs_slice = lhs_section + lidx * lhs.stride(1);
                f64* vwing_slice = verts_wing_j.ptr() + lidx + lidx / colloc_j.shape(0);
                ispc::dkernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwing_slice, verts_wing_j.stride(2), verts_wing_j.stride(1), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
            }).name("wing pass");

            auto last_row = graph.for_each_index(end_wing, colloc_j.stride(2), [=] (i64 lidx) {
                f64* lhs_slice = lhs_section + lidx * lhs.stride(1);
                f64* vwing_slice = verts_wing_j.ptr() + lidx + lidx / colloc_j.shape(0);
                ispc::dkernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwing_slice, verts_wing_j.stride(2), verts_wing_j.stride(1), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
            }).name("last_row");

            auto cond = graph.emplace([=, &condition] {
                return condition[condition_idx] < iteration ? 0 : 1; // 0 means continue, 1 means break (exit loop)
            }).name("condition");
            auto wake_pass = graph.for_each_index(zero, colloc_j.shape(0), [=, &condition] (i64 j) {
                f64* lhs_slice = lhs_section + (j+end_wing) * lhs.stride(1);
                f64* vwake_slice = verts_wake_j.ptr() + verts_wake_j.offset({j, -2-condition[condition_idx], 0});
                ispc::dkernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwake_slice, verts_wake_j.stride(2), verts_wake_j.stride(1), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
            }).name("wake pass");
            auto back = graph.emplace([=, &condition]{
                ++condition[condition_idx];
                return 0; // 0 means continue
            }).name("while back");

            begin.precede(wing_pass, last_row);
            wing_pass.precede(end);
            last_row.precede(cond);
            cond.precede(wake_pass, end); // 0 and 1
            wake_pass.precede(back);
            back.precede(cond);

            offset_i += colloc[m_i].stride(2); // this is assuming contiguous view
        }
        offset_j += colloc[m_j].stride(2);  // this is assuming contiguous view
    }

    // graph.dump(std::cout);
    Executor::get().run(graph).wait();
}

/// @brief Add velocity contributions to the right hand side vector
/// @details
/// Add velocity contributions to the right hand side vector of the VLM system
/// @param rhs right hand side vector
/// @param normals normals of all surfaces
/// @param velocities displacement velocities of all surfaces
void BackendCPU::rhs_assemble_velocities(TensorView1dD& rhs, const MultiTensorView3dD& normals, const MultiTensorView3dD& velocities) {
    // const tiny::ScopedTimer timer("RHS");

    tf::Taskflow taskflow;
    auto end = taskflow.placeholder();

    i64 offset = 0;
    for (i64 m = 0; m < normals.size(); m++) {
        const auto& normals_i = normals[m];
        const auto& velocities_i = velocities[m];
        auto task = taskflow.for_each_index((i64)0, normals_i.shape(1), [=] (i64 j) {
            for (i64 i = 0; i < normals_i.shape(0); i++) {
                const i64 lidx = offset + i + j * normals_i.stride(1);
                rhs(lidx) += - (
                    velocities_i(i, j, 0) * normals_i(i, j, 0) +
                    velocities_i(i, j, 1) * normals_i(i, j, 1) +
                    velocities_i(i, j, 2) * normals_i(i, j, 2));
            }
        });
        task.precede(end);
        offset += normals_i.stride(2);
    }
    Executor::get().run(taskflow).wait();
}

void BackendCPU::rhs_assemble_wake_influence(TensorView1dD& rhs, const MultiTensorView2dD& gamma_wake, const MultiTensorView3dD& colloc, const MultiTensorView3dD& normals, const MultiTensorView3dD& verts_wake, const std::vector<bool>& lifting, i32 iteration) {
    // const tiny::ScopedTimer timer("Wake Influence");
    assert(lifting.size() == normals.size());

    tf::Taskflow taskflow;

    auto begin = taskflow.placeholder();
    auto end = taskflow.placeholder();

    i64 offset = 0;
    for (i32 m_i = 0; m_i < normals.size(); m_i++) {
        const i64 m_i_num_panels = normals[m_i].shape(0) * normals[m_i].shape(1);
        auto wake_influence = taskflow.for_each_index((i64)0, m_i_num_panels, [=, &lifting] (i64 idx) {
            // Loop over the wakes
            for (i32 m_j = 0; m_j < normals.size(); m_j++) {
                if (!lifting[m_j]) continue;
                const auto& normals_i = normals[m_i];
                const auto& colloc_i = colloc[m_i];
                const auto& gamma_wake_j = gamma_wake[m_j];
                const auto& verts_wake_j = verts_wake[m_j];
                ispc::dkernel_wake_influence(
                    colloc_i.ptr() + idx, // technically incorrect as it doesnt consider the edge case of non-contiguous buffer
                    colloc_i.stride(2),
                    normals_i.ptr() + idx, // identical as colloc_i
                    normals_i.stride(2),
                    verts_wake_j.ptr(),
                    verts_wake_j.stride(2),
                    verts_wake_j.shape(1),
                    verts_wake_j.shape(0),
                    gamma_wake_j.ptr(), 
                    rhs.ptr() + offset + idx,
                    sigma_vatistas,
                    iteration
                );
            }
        });

        begin.precede(wake_influence);
        wake_influence.precede(end);
        offset += m_i_num_panels;
    }

    Executor::get().run(taskflow).wait();
}

void BackendCPU::forces_steady(
    const TensorView3dD& verts_wing,
    const TensorView2dD& gamma_delta, // chordwise delta
    const TensorView3dD& velocities,
    const TensorView3dD& forces
) {
    const f64 rho = 1.0f; // TODO: remove hardcoded rho
    for (i64 j = 0; j < gamma_delta.shape(1); j++) { // chordwise
        for (i64 i = 0; i < gamma_delta.shape(0); i++) { // spanwise
            const linalg::double3 V{velocities(i, j, 0), velocities(i, j, 1), velocities(i, j, 2)}; // local velocity (freestream + displacement vel)

            const linalg::double3 vertex0{verts_wing(i, j, 0), verts_wing(i, j, 1), verts_wing(i, j, 2)}; // upper left
            const linalg::double3 vertex1{verts_wing(i+1, j, 0), verts_wing(i+1, j, 1), verts_wing(i+1, j, 2)}; // upper right

            linalg::double3 force = rho * gamma_delta(i, j) * linalg::cross(V, vertex1 - vertex0); // steady contribution

            forces(i, j, 0) = force.x;
            forces(i, j, 1) = force.y;
            forces(i, j, 2) = force.z;
        }
    }
}

void BackendCPU::forces_unsteady(
    const TensorView3dD& verts_wing,
    const TensorView2dD& gamma_delta, // chordwise delta
    const TensorView2dD& dgamma_dt, // dgamma/dt
    const TensorView3dD& velocities,
    const TensorView2dD& areas,
    const TensorView3dD& normals,
    const TensorView3dD& forces
) {
    forces_steady(verts_wing, gamma_delta, velocities, forces); // steady part
    
    const f64 rho = 1.0f; // TODO: remove hardcoded rho
    for (i64 j = 0; j < gamma_delta.shape(1); j++) { // chordwise
        for (i64 i = 0; i < gamma_delta.shape(0); i++) { // spanwise
            const linalg::double3 normal{normals(i, j, 0), normals(i, j, 1), normals(i, j, 2)};
            linalg::double3 force = rho * dgamma_dt(i, j) * areas(i, j) * normal; // unsteady contribution

            forces(i, j, 0) += force.x;
            forces(i, j, 1) += force.y;
            forces(i, j, 2) += force.z;
        }
    }
}

f64 BackendCPU::coeff_cl(
    const TensorView3dD& forces,
    const linalg::double3& lift_axis,
    const linalg::double3& freestream,
    const f64 rho,
    const f64 area
) {
    f64 cl = 0.0f;
    for (i64 j = 0; j < forces.shape(1); j++) {
        for (i64 i = 0; i < forces.shape(0); i++) {
            const linalg::double3 force = {forces(i, j, 0), forces(i, j, 1), forces(i, j, 2)};
            cl += linalg::dot(force, lift_axis);
        }
    }
    return cl / (0.5 * rho * linalg::length2(freestream) * area);
}

linalg::double3 BackendCPU::coeff_cm(
    const TensorView3dD& forces,
    const TensorView3dD& verts_wing,
    const linalg::double3& ref_pt,
    const linalg::double3& freestream,
    const f64 rho,
    const f64 area,
    const f64 mac
) {
    linalg::double3 cm = {0.0f, 0.0f, 0.0f};
    for (i64 j = 0; j < forces.shape(1); j++) {
        for (i64 i = 0; i < forces.shape(0); i++) {
            const linalg::double3 v0 = {verts_wing(i+0, j, 0), verts_wing(i+0, j, 1), verts_wing(i+0, j, 2)}; // left leading vortex line
            const linalg::double3 v1 = {verts_wing(i+1, j, 0), verts_wing(i+1, j, 1), verts_wing(i+1, j, 2)}; // right leading vortex line
            const linalg::double3 force = {forces(i, j, 0), forces(i, j, 1), forces(i, j, 2)};

            const linalg::double3 f_applied = 0.5 * (v0 + v1); // force applied at the center of leading edge vortex line
            const linalg::double3 lever = f_applied - ref_pt;
            cm += linalg::cross(lever, force);
        }
    }
    return cm / (0.5 * rho * linalg::length2(freestream) * area * mac);
}

// TODO: change this to use the per panel local alpha (in global frame)
void BackendCPU::mesh_metrics(const f64 alpha_rad, const MultiTensorView3dD& verts_wing, MultiTensorView3dD& colloc, MultiTensorView3dD& normals, MultiTensorView2dD& areas) {
    // parallel for
    for (int m = 0; m < colloc.size(); m++) {
        auto& colloc_m = colloc[m];
        auto& normals_m = normals[m];
        auto& areas_m = areas[m];
        auto& verts_wing_m = verts_wing[m];
        // parallel for
        for (i64 j = 0; j < colloc_m.shape(1); j++) {
            // inner vectorized loop
            for (i64 i = 0; i < colloc_m.shape(0); i++) {
                const linalg::double3 vertex0{verts_wing_m(i, j, 0), verts_wing_m(i, j, 1), verts_wing_m(i, j, 2)}; // upper left
                const linalg::double3 vertex1{verts_wing_m(i+1, j, 0), verts_wing_m(i+1, j, 1), verts_wing_m(i+1, j, 2)}; // upper right
                const linalg::double3 vertex2{verts_wing_m(i+1, j+1, 0), verts_wing_m(i+1, j+1, 1), verts_wing_m(i+1, j+1, 2)}; // lower right
                const linalg::double3 vertex3{verts_wing_m(i, j+1, 0), verts_wing_m(i, j+1, 1), verts_wing_m(i, j+1, 2)}; // lower left

                const linalg::double3 normal_vec = linalg::normalize(linalg::cross(vertex3 - vertex1, vertex2 - vertex0));
                normals_m(i, j, 0) = normal_vec.x;
                normals_m(i, j, 1) = normal_vec.y;
                normals_m(i, j, 2) = normal_vec.z;

                // 3 vectors f (P0P3), b (P0P2), e (P0P1) to compute the area:
                // area = 0.5 * (||f x b|| + ||b x e||)
                // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
                const linalg::double3 vec_f = vertex3 - vertex0;
                const linalg::double3 vec_b = vertex2 - vertex0;
                const linalg::double3 vec_e = vertex1 - vertex0;

                areas_m(i, j) = 0.5 * (linalg::length(linalg::cross(vec_f, vec_b)) + linalg::length(linalg::cross(vec_b, vec_e)));
                
                // High AoA correction (Aerodynamic Optimization of Aircraft Wings Using a Coupled VLM2.5D RANS Approach) Eq 3.4 p21
                // https://publications.polymtl.ca/2555/1/2017_MatthieuParenteau.pdf
                const f64 factor = (alpha_rad < EPS_f) ? 0.5 : 0.5 * (alpha_rad / (std::sin(alpha_rad) + EPS_f));
                const linalg::double3 chord_vec = 0.5 * (vertex2 + vertex3 - vertex0 - vertex1);
                const linalg::double3 colloc_pt = 0.5 * (vertex0 + vertex1) + factor * chord_vec;

                colloc_m(i, j, 0) = colloc_pt.x;
                colloc_m(i, j, 1) = colloc_pt.y;
                colloc_m(i, j, 2) = colloc_pt.z;
            }
        }
    }
}

/// @brief Computes the mean chord of a set of panels
/// @details
/// Mean Aerodynamic Chord = \frac{2}{S} \int_{0}^{b/2} c(y)^{2} dy
/// Integration using the Trapezoidal Rule
/// Validated against empirical formulas for tapered wings
/// @param j first panel index spanwise
/// @param n number of panels spanwise
/// @return mean chord of the set of panels
f64 BackendCPU::mesh_mac(const TensorView3dD& verts_wing, const TensorView2dD& areas) {
    f64 mac = 0.0f;
    // loop over panel chordwise sections in spanwise direction
    // Note: can be done optimally with vertical fused simd
    for (i64 i = 0; i < areas.shape(0); i++) {
        // left and right chord lengths
        const f64 dx0 = verts_wing(i+0, 0, 0) - verts_wing(i+0, -1, 0);
        const f64 dy0 = verts_wing(i+0, 0, 1) - verts_wing(i+0, -1, 1);
        const f64 dz0 = verts_wing(i+0, 0, 2) - verts_wing(i+0, -1, 2);
        const f64 dx1 = verts_wing(i+1, 0, 0) - verts_wing(i+1, -1, 0);
        const f64 dy1 = verts_wing(i+1, 0, 1) - verts_wing(i+1, -1, 1);
        const f64 dz1 = verts_wing(i+1, 0, 2) - verts_wing(i+1, -1, 2);
        const f64 c0 = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
        const f64 c1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
        // Panel width
        const f64 dx3 = verts_wing(i+1, 0, 0) - verts_wing(i+0, 0, 0);
        const f64 dy3 = verts_wing(i+1, 0, 1) - verts_wing(i+0, 0, 1);
        const f64 dz3 = verts_wing(i+1, 0, 2) - verts_wing(i+0, 0, 2);
        const f64 width = std::sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

        mac += 0.5 * (c0 * c0 + c1 * c1) * width;
    }
    // Since we divide by half the total wing area (both sides) we dont need to multiply by 2

    return mac / sum(areas);
}

void BackendCPU::gamma_wake_from_coeffs(
    const TensorView2dD& gamma_wake,
    const TensorView2dD& gamma_coeffs, // already shifted to the trailing edge
    i32 harmonics,
    f64 tn,
    f64 omega,
    f64 dt,
    i64 iteration
)
{
    assert(gamma_coeffs.shape(0) == gamma_wake.shape(0));

    const i32 unknowns = 2 * harmonics + 1;
    const f64 sqrt_unknowns = 1.f / std::sqrt(static_cast<f64>(unknowns));
    const f64 sqrt_unknowns_2 = std::sqrt(2.f / static_cast<f64>(unknowns));

    tf::Taskflow taskflow;
    auto end = taskflow.placeholder();

    i64 wake_start = gamma_wake.shape(1) - iteration;
    auto task = taskflow.for_each_index(wake_start, gamma_wake.shape(1), [=] (i64 j) {
        for (i64 i = 0; i < gamma_wake.shape(0); i++) { // col
            f64 gamma_w = gamma_coeffs(i, 0) * sqrt_unknowns;
            for (i64 h = 0; h < harmonics; h++) {
                const f64 omega_k = omega * (f64)(h+1);
                gamma_w += gamma_coeffs(i, 2*h+1) * std::cos(omega_k * (tn - (f64)(j - wake_start + 1)*dt)) * sqrt_unknowns_2;
                gamma_w += gamma_coeffs(i, 2*h+2) * std::sin(omega_k * (tn - (f64)(j - wake_start + 1)*dt)) * sqrt_unknowns_2;
            }
            gamma_wake(i, j) = gamma_w;
        }
    });
    task.precede(end);
    Executor::get().run(taskflow).wait();
}

// TODO: replace with BLAS asum ?
f64 BackendCPU::sum(const TensorView1dD& tensor) {
    f64 sum = 0.0;
    for (i64 i = 0; i < tensor.shape(0); i++) {
        sum += tensor(i);
    }
    return sum;
}

// TODO: replace with BLAS asum ?
f64 BackendCPU::sum(const TensorView2dD& tensor) {
    f64 sum = 0.0;
    for (i64 j = 0; j < tensor.shape(1); j++) {
        for (i64 i = 0; i < tensor.shape(0); i++) {
            sum += tensor(i, j);
        }
    }
    return sum;
}
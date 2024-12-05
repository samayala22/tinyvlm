#include "vlm_backend_cpu.hpp"
#include "vlm_backend_cpu_kernels_ispc.h"

#include "tinytimer.hpp"
#include "tinycpuid2.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_executor.hpp" // includes taskflow/taskflow.hpp

#include <algorithm> // std::fill
#include <cstdio> // std::printf
#include <thread> // std::hardware_concurrency()

#include <taskflow/algorithm/for_each.hpp>

#include <lapacke.h>
#include <cblas.h>

using namespace vlm;
using namespace linalg::ostream_overloads;

class CPU_Kernels;

/// @brief Memory manager implementation for the CPU backend
class CPU_Memory final : public Memory {
    public:
        explicit CPU_Memory() : Memory(true) {}
        ~CPU_Memory() override= default;
        void* alloc(Location location, i64 size_bytes) const override {return std::malloc(size_bytes);}
        void free(Location location, void* ptr) const override {std::free(ptr);}
        void copy(Location dst_loc, void* dst, i64 dst_stride, Location src_loc, const void* src, i64 src_stride, i64 elem_size, i64 size) const override {
            if (dst_stride == 1 && src_stride == 1) {
                std::memcpy(dst, src, size * elem_size);
            } else {
                const char* src_ptr = static_cast<const char*>(src);
                char* dst_ptr = static_cast<char*>(dst);
                i64 byte_stride = src_stride * elem_size;
                i64 byte_dst_stride = dst_stride * elem_size;

                for (i64 i = 0; i < size; ++i) {
                    std::memcpy(dst_ptr + i * byte_dst_stride, src_ptr + i * byte_stride, elem_size);
                }
            }
        }
        void fill(Location location, float* ptr, i64 stride, float value, i64 size) const override {
            if (stride == 1) {
                std::fill(ptr, ptr + size, value);
            } else {
                for (i64 i = 0; i < size; i++) {
                    ptr[i * stride] = value;
                }
            }
        }
        void fill(Location location, double* ptr, i64 stride, double value, i64 size) const override {
            if (stride == 1) {
                std::fill(ptr, ptr + size, value);
            } else {
                for (i64 i = 0; i < size; i++) {
                    ptr[i * stride] = value;
                }
            }
        }
};

void print_cpu_info() {
    tiny::CPUID2 cpuid;
    std::printf("DEVICE: %s (%d threads)\n", cpuid.full_name.c_str(), std::thread::hardware_concurrency());
}

class CPU_LU final : public LU {
    public:
        explicit CPU_LU(std::unique_ptr<Memory> memory);
        ~CPU_LU() override = default;
        
        void init(const TensorView<f32, 2, Location::Device>& A) override;
        void factorize(const TensorView<f32, 2, Location::Device>& A) override;
        void solve(const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& x) override;

    private:
        Tensor<i32, 1, Location::Device> ipiv{m_memory.get()};
};

CPU_LU::CPU_LU(std::unique_ptr<Memory> memory) : LU(std::move(memory)) {}

void CPU_LU::init(const TensorView<f32, 2, Location::Device>& A) {
    ipiv.init({A.shape(0)}); // row pivoting
}

void CPU_LU::factorize(const TensorView<f32, 2, Location::Device>& A) {
    assert(ipiv.view().shape(0) == A.shape(0));
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, A.shape(0), A.shape(1), A.ptr(), A.stride(1), ipiv.ptr());
}

void CPU_LU::solve(const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& x) {
    LAPACKE_sgetrs(
        LAPACK_COL_MAJOR,
        'N',
        A.shape(1),
        x.shape(1),
        A.ptr(),
        A.stride(1),
        ipiv.ptr(),
        x.ptr(),
        x.stride(1)
    );
}

class CPU_BLAS final : public BLAS {
    public:
        explicit CPU_BLAS() = default;
        ~CPU_BLAS() override= default;

        void gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, const TensorView<f32, 1, Location::Device>& y, Trans trans = Trans::No) override;
        void gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, const TensorView<f32, 2, Location::Device>& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) override;
        void axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, const TensorView<f32, 1, Location::Device>& y) override;
        void axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, const TensorView<f32, 2, Location::Device>& y) override;
        f32 norm(const TensorView<f32, 1, Location::Device>& x) override;
};

f32 CPU_BLAS::norm(const TensorView<f32, 1, Location::Device>& x) {
    return cblas_snrm2(x.shape(0), x.ptr(), x.stride(0));
}

CBLAS_TRANSPOSE trans_to_cblas(BLAS::Trans trans) {
    switch (trans) {
        case BLAS::Trans::No: return CblasNoTrans;
        case BLAS::Trans::Yes: return CblasTrans;
    }
}

void CPU_BLAS::gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, const TensorView<f32, 1, Location::Device>& y, Trans trans) {
    assert(A.stride(0) == 1);

    i32 m = (trans == BLAS::Trans::No) ? A.shape(0) : A.shape(1);
    i32 n = (trans == BLAS::Trans::No) ? A.shape(1) : A.shape(0);

    cblas_sgemv(
        CblasColMajor,
        trans_to_cblas(trans),
        m,
        n,
        alpha,
        A.ptr(),
        static_cast<int>(A.stride(1)),
        x.ptr(),
        static_cast<int>(x.stride(0)),
        beta,
        y.ptr(),
        static_cast<int>(y.stride(0))
    );
}

void CPU_BLAS::gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, const TensorView<f32, 2, Location::Device>& C, Trans trans_a, Trans trans_b) {
    assert(A.stride(0) == 1);
    assert(B.stride(0) == 1);
    assert(C.stride(0) == 1);

    i32 m = (trans_a == BLAS::Trans::No) ? A.shape(0) : A.shape(1);
    i32 n = (trans_b == BLAS::Trans::No) ? B.shape(1) : B.shape(0);
    i32 k = (trans_a == BLAS::Trans::No) ? A.shape(1) : A.shape(0);

    cblas_sgemm(
        CblasColMajor,
        trans_to_cblas(trans_a),
        trans_to_cblas(trans_b),
        m,
        n,
        k,
        alpha,
        A.ptr(),
        static_cast<int>(A.stride(1)),
        B.ptr(),
        static_cast<int>(B.stride(1)),
        beta,
        C.ptr(),
        static_cast<int>(C.stride(1))
    );
}

void CPU_BLAS::axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, const TensorView<f32, 1, Location::Device>& y) {
    cblas_saxpy(
        x.shape(0),
        alpha, 
        x.ptr(),
        x.stride(0),
        y.ptr(),
        y.stride(0)
    );
}

void CPU_BLAS::axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, const TensorView<f32, 2, Location::Device>& y) {
    assert(x.shape() == y.shape());
    f32* y_ptr = y.ptr();
    f32* x_ptr = x.ptr();
    for (i64 j = 0; j < x.shape(1); j++) {
        for (i64 i = 0; i < x.shape(0); i++) {
            y_ptr[j * y.stride(1) + i] += alpha * x_ptr[j * x.stride(1) + i];
        }
    }
}

BackendCPU::BackendCPU() : Backend(create_memory_manager(), create_blas()) {
    print_cpu_info();
}

BackendCPU::~BackendCPU() {}

std::unique_ptr<Memory> BackendCPU::create_memory_manager() { return std::make_unique<CPU_Memory>(); }
// std::unique_ptr<Kernels> create_kernels() { return std::make_unique<CPU_Kernels>(); }
std::unique_ptr<LU> BackendCPU::create_lu_solver() { return std::make_unique<CPU_LU>(create_memory_manager()); }
std::unique_ptr<BLAS> BackendCPU::create_blas() { return std::make_unique<CPU_BLAS>(); }


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
void BackendCPU::lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView3D<Location::Device>& verts_wake, std::vector<i32>& condition, i32 iteration) {
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

            f32* lhs_section = lhs.ptr() + offset_i + offset_j * lhs.stride(1);
            
            const i64 zero = 0;
            const i64 end_wing = (colloc_j.shape(1) - 1) * colloc_j.shape(0);
            
            auto wing_pass = graph.for_each_index(zero, end_wing, [=] (i64 lidx) {
                f32* lhs_slice = lhs_section + lidx * lhs.stride(1);
                f32* vwing_slice = verts_wing_j.ptr() + lidx + lidx / colloc_j.shape(0);
                ispc::kernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwing_slice, verts_wing_j.stride(2), verts_wing_j.stride(1), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
            }).name("wing pass");

            auto last_row = graph.for_each_index(end_wing, colloc_j.stride(2), [=] (i64 lidx) {
                f32* lhs_slice = lhs_section + lidx * lhs.stride(1);
                f32* vwing_slice = verts_wing_j.ptr() + lidx + lidx / colloc_j.shape(0);
                ispc::kernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwing_slice, verts_wing_j.stride(2), verts_wing_j.stride(1), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
            }).name("last_row");

            auto cond = graph.emplace([=, &condition] {
                return condition[condition_idx] < iteration ? 0 : 1; // 0 means continue, 1 means break (exit loop)
            }).name("condition");
            auto wake_pass = graph.for_each_index(zero, colloc_j.shape(0), [=, &condition] (i64 j) {
                f32* lhs_slice = lhs_section + (j+end_wing) * lhs.stride(1);
                f32* vwake_slice = verts_wake_j.ptr() + verts_wake_j.offset({j, -2-condition[condition_idx], 0});
                ispc::kernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwake_slice, verts_wake_j.stride(2), verts_wake_j.stride(1), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
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
void BackendCPU::rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& velocities) {
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

void BackendCPU::rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView2D<Location::Device>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& verts_wake, i32 iteration) {
    // const tiny::ScopedTimer timer("Wake Influence");

    tf::Taskflow taskflow;

    auto begin = taskflow.placeholder();
    auto end = taskflow.placeholder();

    auto wake_influence = taskflow.for_each_index((i64)0, rhs.size(), [&] (i64 idx) {
        for (i32 i = 0; i < normals.size(); i++) {
            const auto& normals_i = normals[i];
            const auto& colloc_i = colloc[i];
            const auto& gamma_wake_i = gamma_wake[i];
            const auto& verts_wake_i = verts_wake[i];
            ispc::kernel_wake_influence(
                colloc_i.ptr() + idx,
                colloc_i.stride(2),
                normals_i.ptr() + idx,
                normals_i.stride(2),
                verts_wake_i.ptr(),
                verts_wake_i.stride(2),
                verts_wake_i.shape(1),
                verts_wake_i.shape(0),
                gamma_wake_i.ptr(), 
                rhs.ptr() + idx,
                sigma_vatistas,
                iteration
            );
        }
    });

    begin.precede(wake_influence);
    wake_influence.precede(end);

    Executor::get().run(taskflow).wait();
}

// TODO: reactivate this function taking care to take into account the multibody interactions
// We should be passing two different pointers of the influenced wake vertex and the start of the influencing wake vertices
void BackendCPU::displace_wake_rollup(MultiTensorView3D<Location::Device>& wake_rollup, const MultiTensorView3D<Location::Device>& verts_wake, const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wake, f32 dt, i32 iteration) {
    // // const tiny::ScopedTimer timer("Wake Rollup");
    // tf::Taskflow taskflow;

    // auto begin = taskflow.placeholder();
    // auto end = taskflow.placeholder();

    // for (i64 m = 0; m < gamma_wing.size(); m++) {
    //     auto& gamma_wing_m = gamma_wing[m];
    //     auto& gamma_wake_m = gamma_wake[m];
    //     auto& verts_wake_m = verts_wake[m];
    //     const i64 wake_begin = (verts_wake_m.shape(1) - iteration) * (verts_wake_m.shape(0));
    //     const i64 wake_end = verts_wake_m.shape(1) * (verts_wake_m.shape(0));
    //     auto rollup = taskflow.for_each_index(wake_begin, wake_end, [&] (i64 vidx) {
    //         for (i64 i = 0; i < verts_wake.size(); i++) {
    //             ispc::kernel_rollup(
    //                 verts_wake.ptr + verts_wake.layout.offset(i),
    //                 verts_wake.layout.stride(),
    //                 verts_wake.layout.nc(i),
    //                 verts_wake.layout.ns(i),
    //                 vidx,
    //                 verts_wing.ptr + verts_wing.layout.offset(i),
    //                 verts_wing.layout.stride(),
    //                 verts_wing.layout.nc(i),
    //                 verts_wing.layout.ns(i),
    //                 wake_rollup.ptr + wake_rollup.layout.offset(i),
    //                 wake_rollup.layout.stride(),
    //                 gamma_wing_i.ptr(),
    //                 gamma_wake_i.ptr(),
    //                 sigma_vatistas,
    //                 dt,
    //                 iteration
    //             );
    //         }
    //     });
    //     auto copy = taskflow.emplace([&, wake_begin, wake_end, m]{
    //         memory->copy(Location::Device, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 0*verts_wake.layout.stride(), 1, Location::Device, wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 0*wake_rollup.layout.stride(), 1, sizeof(f32), (wake_end - wake_begin));
    //         memory->copy(Location::Device, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 1*verts_wake.layout.stride(), 1, Location::Device, wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 1*wake_rollup.layout.stride(), 1, sizeof(f32), (wake_end - wake_begin));
    //         memory->copy(Location::Device, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 2*verts_wake.layout.stride(), 1, Location::Device, wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 2*wake_rollup.layout.stride(), 1, sizeof(f32), (wake_end - wake_begin));
    //     });
    //     begin.precede(rollup);
    //     rollup.precede(copy);
    //     copy.precede(end);
    // }

    // Executor::get().run(taskflow).wait();
}

f32 BackendCPU::coeff_steady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const FlowData& flow, f32 area) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cl = 0.0f;

    for (i64 j = 0; j < gamma_delta.shape(1); j++) {
        for (i64 i = 0; i < gamma_delta.shape(0); i++) {
            const linalg::float3 vertex0{verts_wing(i, j, 0), verts_wing(i, j, 1), verts_wing(i, j, 2)}; // upper left
            const linalg::float3 vertex1{verts_wing(i+1, j, 0), verts_wing(i+1, j, 1), verts_wing(i+1, j, 2)}; // upper right

            // Leading edge vector pointing outward from wing root
            const linalg::float3 dl = vertex1 - vertex0;
            // const linalg::float3 local_left_chord = linalg::normalize(v3 - v0);
            // const linalg::float3 projected_vector = linalg::dot(dl, local_left_chord) * local_left_chord;
            // dl -= projected_vector; // orthogonal leading edge vector
            // Distance from the center of leading edge to the reference point
            const linalg::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * gamma_delta(i, j);
            cl += linalg::dot(force, flow.lift_axis); // projection on the body lift axis
        }
    }
    cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;

    return cl;
}

void BackendCPU::forces_unsteady(
    const TensorView3D<Location::Device>& verts_wing,
    const TensorView2D<Location::Device>& gamma_delta,
    const TensorView2D<Location::Device>& gamma,
    const TensorView2D<Location::Device>& gamma_prev,
    const TensorView3D<Location::Device>& velocities,
    const TensorView2D<Location::Device>& areas,
    const TensorView3D<Location::Device>& normals,
    const TensorView3D<Location::Device>& forces,
    f32 dt
    ) {
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    for (i64 j = 0; j < gamma_delta.shape(1); j++) {
        for (i64 i = 0; i < gamma_delta.shape(0); i++) {

            const linalg::float3 V{velocities(i, j, 0), velocities(i, j, 1), velocities(i, j, 2)}; // local velocity (freestream + displacement vel)

            const linalg::float3 vertex0{verts_wing(i, j, 0), verts_wing(i, j, 1), verts_wing(i, j, 2)}; // upper left
            const linalg::float3 vertex1{verts_wing(i+1, j, 0), verts_wing(i+1, j, 1), verts_wing(i+1, j, 2)}; // upper right
            const linalg::float3 normal{normals(i, j, 0), normals(i, j, 1), normals(i, j, 2)};

            linalg::float3 force = {0.0f, 0.0f, 0.0f};
            const f32 gamma_dt = (gamma(i, j) - gamma_prev(i, j)) / dt; // backward difference

            // Joukowski method
            force += rho * gamma_delta(i, j) * linalg::cross(V, vertex1 - vertex0); // steady contribution
            force += rho * gamma_dt * areas(i, j) * normal; // unsteady contribution

            forces(i, j, 0) = force.x;
            forces(i, j, 1) = force.y;
            forces(i, j, 2) = force.z;
        }
    }
}

f32 BackendCPU::coeff_cl(
    const TensorView3D<Location::Device>& forces,
    const linalg::float3& lift_axis,
    const linalg::float3& freestream,
    const f32 rho,
    const f32 area
) {
    f32 cl = 0.0f;
    for (i64 j = 0; j < forces.shape(1); j++) {
        for (i64 i = 0; i < forces.shape(0); i++) {
            const linalg::float3 force = {forces(i, j, 0), forces(i, j, 1), forces(i, j, 2)};
            cl += linalg::dot(force, lift_axis);
        }
    }
    return cl / (0.5f * rho * linalg::length2(freestream) * area);
}

linalg::float3 BackendCPU::coeff_cm(
    const TensorView3D<Location::Device>& forces,
    const TensorView3D<Location::Device>& verts_wing,
    const linalg::float3& ref_pt,
    const linalg::float3& freestream,
    const f32 rho,
    const f32 area,
    const f32 mac
) {
    linalg::float3 cm = {0.0f, 0.0f, 0.0f};
    for (i64 j = 0; j < forces.shape(1); j++) {
        for (i64 i = 0; i < forces.shape(0); i++) {
            const linalg::float3 v0 = {verts_wing(i+0, j, 0), verts_wing(i+0, j, 1), verts_wing(i+0, j, 2)}; // left leading vortex line
            const linalg::float3 v1 = {verts_wing(i+1, j, 0), verts_wing(i+1, j, 1), verts_wing(i+1, j, 2)}; // right leading vortex line
            const linalg::float3 force = {forces(i, j, 0), forces(i, j, 1), forces(i, j, 2)};

            // const linalg::float3 dst_to_ref = ref_pt - 0.5f * (v0 + v1);
            // cm += linalg::cross(force, dst_to_ref);

            const linalg::float3 f_applied = 0.5f * (v0 + v1);
            const linalg::float3 lever = f_applied - ref_pt;
            cm += linalg::cross(lever, force);
        }
    }
    return cm / (0.5f * rho * linalg::length2(freestream) * area * mac);
}

f32 BackendCPU::coeff_steady_cd_single(const TensorView3D<Location::Device>& verts_wake, const TensorView2D<Location::Device>& gamma_wake, const FlowData& flow, f32 area) {
    // tiny::ScopedTimer timer("Compute CD");
    f32 cd = ispc::kernel_trefftz_cd(verts_wake.ptr(), verts_wake.stride(2), verts_wake.shape(1), verts_wake.shape(0), gamma_wake.ptr(), sigma_vatistas);
    cd /= linalg::length2(flow.freestream) * area;
    return cd;
}

// Using Trefftz plane
// f32 BackendCPU::compute_coefficient_cl(
//     const FlowData& flow,
//     const f32 area,
//     const i64 j,
//     const i64 n) 
// {
//     Mesh& m = mesh;
//     ispc::MeshProxy mesh_proxy = {
//         m.ns, m.nc, m.nb_panels_wing(),
//         {m.v.x.data(), m.v.y.data(), m.v.z.data()}, 
//         {m.colloc.x.data(), m.colloc.y.data(), m.colloc.z.data()},
//         {m.normal.x.data(), m.normal.y.data(), m.normal.z.data()}
//     };
//     f32 cl = ispc::kernel_trefftz_cl(mesh_proxy, gamma.data(), j, n);
//     cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;
//     return cl;
// }

// TODO: change this to use the per panel local alpha (in global frame)
void BackendCPU::mesh_metrics(const f32 alpha_rad, const MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, MultiTensorView2D<Location::Device>& areas) {
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
                const linalg::float3 vertex0{verts_wing_m(i, j, 0), verts_wing_m(i, j, 1), verts_wing_m(i, j, 2)}; // upper left
                const linalg::float3 vertex1{verts_wing_m(i+1, j, 0), verts_wing_m(i+1, j, 1), verts_wing_m(i+1, j, 2)}; // upper right
                const linalg::float3 vertex2{verts_wing_m(i+1, j+1, 0), verts_wing_m(i+1, j+1, 1), verts_wing_m(i+1, j+1, 2)}; // lower right
                const linalg::float3 vertex3{verts_wing_m(i, j+1, 0), verts_wing_m(i, j+1, 1), verts_wing_m(i, j+1, 2)}; // lower left

                const linalg::float3 normal_vec = linalg::normalize(linalg::cross(vertex3 - vertex1, vertex2 - vertex0));
                normals_m(i, j, 0) = normal_vec.x;
                normals_m(i, j, 1) = normal_vec.y;
                normals_m(i, j, 2) = normal_vec.z;

                // 3 vectors f (P0P3), b (P0P2), e (P0P1) to compute the area:
                // area = 0.5 * (||f x b|| + ||b x e||)
                // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
                const linalg::float3 vec_f = vertex3 - vertex0;
                const linalg::float3 vec_b = vertex2 - vertex0;
                const linalg::float3 vec_e = vertex1 - vertex0;

                areas_m(i, j) = 0.5f * (linalg::length(linalg::cross(vec_f, vec_b)) + linalg::length(linalg::cross(vec_b, vec_e)));
                
                // High AoA correction (Aerodynamic Optimization of Aircraft Wings Using a Coupled VLM2.5D RANS Approach) Eq 3.4 p21
                // https://publications.polymtl.ca/2555/1/2017_MatthieuParenteau.pdf
                const f32 factor = (alpha_rad < EPS_f) ? 0.5f : 0.5f * (alpha_rad / (std::sin(alpha_rad) + EPS_f));
                const linalg::float3 chord_vec = 0.5f * (vertex2 + vertex3 - vertex0 - vertex1);
                const linalg::float3 colloc_pt = 0.5f * (vertex0 + vertex1) + factor * chord_vec;

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
f32 BackendCPU::mesh_mac(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& areas) {
    f32 mac = 0.0f;
    // loop over panel chordwise sections in spanwise direction
    // Note: can be done optimally with vertical fused simd
    for (i64 i = 0; i < areas.shape(0); i++) {
        // left and right chord lengths
        const f32 dx0 = verts_wing(i+0, 0, 0) - verts_wing(i+0, -1, 0);
        const f32 dy0 = verts_wing(i+0, 0, 1) - verts_wing(i+0, -1, 1);
        const f32 dz0 = verts_wing(i+0, 0, 2) - verts_wing(i+0, -1, 2);
        const f32 dx1 = verts_wing(i+1, 0, 0) - verts_wing(i+1, -1, 0);
        const f32 dy1 = verts_wing(i+1, 0, 1) - verts_wing(i+1, -1, 1);
        const f32 dz1 = verts_wing(i+1, 0, 2) - verts_wing(i+1, -1, 2);
        const f32 c0 = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
        const f32 c1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
        // Panel width
        const f32 dx3 = verts_wing(i+1, 0, 0) - verts_wing(i+0, 0, 0);
        const f32 dy3 = verts_wing(i+1, 0, 1) - verts_wing(i+0, 0, 1);
        const f32 dz3 = verts_wing(i+1, 0, 2) - verts_wing(i+0, 0, 2);
        const f32 width = std::sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

        mac += 0.5f * (c0 * c0 + c1 * c1) * width;
    }
    // Since we divide by half the total wing area (both sides) we dont need to multiply by 2

    return mac / sum(areas);
}

// TODO: parallelize
f32 BackendCPU::sum(const TensorView1D<Location::Device>& tensor) {
    f32 sum = 0.0f;
    for (i64 i = 0; i < tensor.shape(0); i++) {
        sum += tensor(i);
    }
    return sum;
}

// TODO: parallelize
f32 BackendCPU::sum(const TensorView2D<Location::Device>& tensor) {
    f32 sum = 0.0f;
    for (i64 j = 0; j < tensor.shape(1); j++) {
        for (i64 i = 0; i < tensor.shape(0); i++) {
            sum += tensor(i, j);
        }
    }
    return sum;
}
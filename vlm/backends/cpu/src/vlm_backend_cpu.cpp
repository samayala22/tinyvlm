#include "vlm_backend_cpu.hpp"
#include "vlm_backend_cpu_kernels_ispc.h"

#include "tinytimer.hpp"
#include "tinycpuid2.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_executor.hpp" // includes taskflow/taskflow.hpp

#include <algorithm> // std::fill
#include <iostream> // std::cout
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
        CPU_Memory() : Memory(true) {}
        ~CPU_Memory() = default;
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
        CPU_LU(std::unique_ptr<Memory> memory);
        ~CPU_LU() = default;
        
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
        CPU_BLAS() = default;
        ~CPU_BLAS() = default;

        void gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, TensorView<f32, 1, Location::Device>& y, Trans trans = Trans::No) override;
        void gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, TensorView<f32, 2, Location::Device>& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) override;
        void axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, TensorView<f32, 1, Location::Device>& y) override;
        void axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, TensorView<f32, 2, Location::Device>& y) override;
};

CBLAS_TRANSPOSE trans_to_cblas(BLAS::Trans trans) {
    switch (trans) {
        case BLAS::Trans::No: return CblasNoTrans;
        case BLAS::Trans::Yes: return CblasTrans;
    }
}

void CPU_BLAS::gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, TensorView<f32, 1, Location::Device>& y, Trans trans) {
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

void CPU_BLAS::gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, TensorView<f32, 2, Location::Device>& C, Trans trans_a, Trans trans_b) {
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

void CPU_BLAS::axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, TensorView<f32, 1, Location::Device>& y) {
    cblas_saxpy(
        x.shape(0),
        alpha, 
        x.ptr(),
        x.stride(0),
        y.ptr(),
        y.stride(0)
    );
}

void CPU_BLAS::axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, TensorView<f32, 2, Location::Device>& y) {
    assert(x.shape() == y.shape());
    f32* y_ptr = y.ptr();
    f32* x_ptr = x.ptr();
    for (i64 j = 0; j < x.shape(1); j++) {
        for (i64 i = 0; i < x.shape(0); i++) {
            y_ptr[j * y.stride(1) + i] += alpha * x_ptr[j * x.stride(1) + i];
        }
    }
}

BackendCPU::BackendCPU() : Backend(std::make_unique<CPU_Memory>()) {
    print_cpu_info();
}

BackendCPU::~BackendCPU() {}

/// @brief Compute the gamma_delta vector
/// @details
/// Compute the gamma_delta vector of the VLM system (\Delta\Gamma = \Gamma_{i,j} - \Gamma_{i-1,j})
/// The vector is computed for each lifting surface of the system
/// @param gamma_delta gamma_delta vector
/// @param gamma gamma vector
void BackendCPU::gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) {
    assert(gamma_delta.layout.dims() == 1);
    assert(gamma.layout.dims() == 1);
    
    tf::Taskflow graph;

    auto begin = graph.placeholder();
    auto end = graph.placeholder();

    for (const auto& surf : gamma_delta.layout.surfaces())  {
        f32* s_gamma_delta = gamma_delta.ptr + surf.offset;
        const f32* s_gamma = gamma.ptr + surf.offset;
        tf::Task first_row = graph.emplace([=]{
            memory->copy(Location::Device, s_gamma_delta, 1, Location::Device, s_gamma, 1, sizeof(*s_gamma_delta), surf.ns);
        });
        tf::Task remaining_rows = graph.for_each_index((i64)1,surf.nc, [=] (i64 b, i64 e) {
            for (i64 i = b; i < e; i++) {
                for (i64 j = 0; j < surf.ns; j++) {
                    s_gamma_delta[i*surf.ns + j] = s_gamma[i*surf.ns + j] - s_gamma[(i-1)*surf.ns + j];
                }
            }
        });

        begin.precede(first_row);
        begin.precede(remaining_rows);
        first_row.precede(end);
        remaining_rows.precede(end);
    };

    Executor::get().run(graph).wait();
    // graph.dump(std::cout);
}

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
void BackendCPU::lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<i32>& condition, i32 iteration) {
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
            auto colloc_i = colloc[m_i];
            auto colloc_j = colloc[m_j];
            auto normals_i = normals[m_i];

            f32* lhs_section = lhs.ptr() + offset_i + offset_j * lhs.stride(1);
            f32* vwing_section = verts_wing.ptr + verts_wing.layout.offset(m_j);
            f32* vwake_section = verts_wake.ptr + verts_wake.layout.offset(m_j);

            const i64 zero = 0;
            const i64 end_wing = (colloc_j.shape(1) - 1) * colloc_j.shape(0);
            
            auto wing_pass = graph.for_each_index(zero, end_wing, [=] (i64 lidx) {
                f32* lhs_slice = lhs_section + lidx * lhs.stride(1);
                f32* vwing_slice = vwing_section + lidx + lidx / colloc_j.shape(0);
                ispc::kernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwing_slice, verts_wing.layout.stride(), verts_wing.layout.ns(m_j), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
            }).name("wing pass");

            auto last_row = graph.for_each_index(end_wing, colloc_j.stride(2), [=] (i64 lidx) {
                f32* lhs_slice = lhs_section + lidx * lhs.stride(1);
                f32* vwing_slice = vwing_section + lidx + lidx / colloc_j.shape(0);
                ispc::kernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwing_slice, verts_wing.layout.stride(), verts_wing.layout.ns(m_j), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
            }).name("last_row");

            auto cond = graph.emplace([=, &condition] {
                return condition[condition_idx] < iteration ? 0 : 1; // 0 means continue, 1 means break (exit loop)
            }).name("condition");
            auto wake_pass = graph.for_each_index(zero, colloc_j.shape(0), [=, &condition] (i64 j) {
                f32* lhs_slice = lhs_section + (j+end_wing) * lhs.stride(1);
                f32* vwake_slice = vwake_section + (verts_wake.layout.nc(m_j) - condition[condition_idx] - 2) * verts_wake.layout.ns(m_j) + j;
                ispc::kernel_influence(colloc_i.stride(2), lhs_slice, colloc_i.ptr(), colloc_i.stride(2), vwake_slice, verts_wake.layout.stride(), verts_wake.layout.ns(m_j), normals_i.ptr(), normals_i.stride(2), sigma_vatistas);
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
void BackendCPU::rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& velocities) {
    // const tiny::ScopedTimer timer("RHS");
    assert(rhs.size() == rhs.size()); // single dim
    assert(rhs.size() == velocities.layout.stride());

    tf::Taskflow taskflow;
    auto end = taskflow.placeholder();

    i64 offset = 0;
    for (i64 m = 0; m < normals.size(); m++) {
        auto& normals_i = normals[m];
        f32* normals_ptr = normals_i.ptr();
        auto task = taskflow.for_each_index((i64)0, normals_i.stride(2), [&, normals_ptr, offset] (i64 i) {
            i64 lidx = offset + i;
            *(rhs.ptr() + lidx) += - (
                velocities[lidx + 0 * velocities.layout.stride()] * normals_ptr[i + 0 * normals_i.stride(2)] +
                velocities[lidx + 1 * velocities.layout.stride()] * normals_ptr[i + 1 * normals_i.stride(2)] +
                velocities[lidx + 2 * velocities.layout.stride()] * normals_ptr[i + 2 * normals_i.stride(2)]);
        });
        task.precede(end);
        offset += normals_i.stride(2);
    }
    Executor::get().run(taskflow).wait();
}

void BackendCPU::rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const View<f32, MultiSurface>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wake, i32 iteration) {
    // const tiny::ScopedTimer timer("Wake Influence");

    tf::Taskflow taskflow;

    auto begin = taskflow.placeholder();
    auto end = taskflow.placeholder();

    auto wake_influence = taskflow.for_each_index((i64)0, (i64)rhs.size(), [&] (i64 idx) {
        for (i32 i = 0; i < normals.size(); i++) {
            auto& normals_i = normals[i];
            auto& colloc_i = colloc[i];
            ispc::kernel_wake_influence(
                colloc_i.ptr() + idx,
                colloc_i.stride(2),
                normals_i.ptr() + idx,
                normals_i.stride(2),
                verts_wake.ptr + verts_wake.layout.offset(i),
                verts_wake.layout.stride(),
                verts_wake.layout.nc(i),
                verts_wake.layout.ns(i),
                gamma_wake.ptr + gamma_wake.layout.offset(i), 
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

void BackendCPU::displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, i32 iteration) {
    // const tiny::ScopedTimer timer("Wake Rollup");
    tf::Taskflow taskflow;

    auto begin = taskflow.placeholder();
    auto end = taskflow.placeholder();

    for (i64 m = 0; m < verts_wake.layout.surfaces().size(); m++) {
        const i64 wake_begin = (verts_wake.layout.nc(m) - iteration) * (verts_wake.layout.ns(m));
        const i64 wake_end = verts_wake.layout.nc(m) * (verts_wake.layout.ns(m));
        auto rollup = taskflow.for_each_index(wake_begin, wake_end, [&] (i64 vidx) {
            for (i64 i = 0; i < verts_wake.layout.surfaces().size(); i++) {
                ispc::kernel_rollup(verts_wake.ptr + verts_wake.layout.offset(i), verts_wake.layout.stride(), verts_wake.layout.nc(i), verts_wake.layout.ns(i), vidx, verts_wing.ptr + verts_wing.layout.offset(i), verts_wing.layout.stride(), verts_wing.layout.nc(i), verts_wing.layout.ns(i), wake_rollup.ptr + wake_rollup.layout.offset(i), wake_rollup.layout.stride(), gamma_wing.ptr + gamma_wing.layout.offset(i), gamma_wake.ptr + gamma_wake.layout.offset(i), sigma_vatistas, dt, iteration);
            }
        });
        auto copy = taskflow.emplace([&, wake_begin, wake_end, m]{
            memory->copy(Location::Device, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 0*verts_wake.layout.stride(), 1, Location::Device, wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 0*wake_rollup.layout.stride(), 1, sizeof(f32), (wake_end - wake_begin));
            memory->copy(Location::Device, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 1*verts_wake.layout.stride(), 1, Location::Device, wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 1*wake_rollup.layout.stride(), 1, sizeof(f32), (wake_end - wake_begin));
            memory->copy(Location::Device, verts_wake.ptr + verts_wake.layout.offset(m) + wake_begin + 2*verts_wake.layout.stride(), 1, Location::Device, wake_rollup.ptr + wake_rollup.layout.offset(m) + wake_begin + 2*wake_rollup.layout.stride(), 1, sizeof(f32), (wake_end - wake_begin));
        });
        begin.precede(rollup);
        rollup.precede(copy);
        copy.precede(end);
    }

    Executor::get().run(taskflow).wait();
}

void BackendCPU::gamma_shed(View<f32, MultiSurface>& gamma_wing, View<f32, MultiSurface>& gamma_wing_prev, View<f32, MultiSurface>& gamma_wake, i32 iteration) {
    // const tiny::ScopedTimer timer("Shed Gamma");

    memory->copy(Location::Device, gamma_wing_prev.ptr, 1, Location::Device, gamma_wing.ptr, 1, sizeof(f32), gamma_wing.size());
    for (i64 i = 0; i < gamma_wake.layout.surfaces().size(); i++) {
        assert(iteration < gamma_wake.layout.nc(i));
        f32* gamma_wake_ptr = gamma_wake.ptr + gamma_wake.layout.offset(i) + (gamma_wake.layout.nc(i) - iteration - 1) * gamma_wake.layout.ns(i);
        f32* gamma_wing_ptr = gamma_wing.ptr + gamma_wing.layout.offset(i) + (gamma_wing.layout.nc(i) - 1) * gamma_wing.layout.ns(i); // last row
        memory->copy(Location::Device, gamma_wake_ptr, 1, Location::Device, gamma_wing_ptr, 1, sizeof(f32), gamma_wing.layout.ns(i));
    }
}

f32 BackendCPU::coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow, f32 area) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cl = 0.0f;

    const i64 nc = gamma_delta.layout.nc();
    const i64 ns = gamma_delta.layout.ns();
    for (i64 i = 0; i < nc; i++) {
        for (i64 j = 0; j < ns; j++) {
            const i64 v0 = (i+0) * verts_wing.layout.ld() + j;
            const i64 v1 = (i+0) * verts_wing.layout.ld() + j + 1;
            const linalg::alias::float3 vertex0{verts_wing.ptr[0*verts_wing.layout.stride() + v0], verts_wing.ptr[1*verts_wing.layout.stride() + v0], verts_wing.ptr[2*verts_wing.layout.stride() + v0]}; // upper left
            const linalg::alias::float3 vertex1{verts_wing.ptr[0*verts_wing.layout.stride() + v1], verts_wing.ptr[1*verts_wing.layout.stride() + v1], verts_wing.ptr[2*verts_wing.layout.stride() + v1]}; // upper right
            // const linalg::alias::float3 v3 = mesh.get_v3(li);
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = vertex1 - vertex0;
            // const linalg::alias::float3 local_left_chord = linalg::normalize(v3 - v0);
            // const linalg::alias::float3 projected_vector = linalg::dot(dl, local_left_chord) * local_left_chord;
            // dl -= projected_vector; // orthogonal leading edge vector
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * gamma_delta.ptr[i * gamma_delta.layout.ld() + j];
            cl += linalg::dot(force, flow.lift_axis); // projection on the body lift axis
        }
    }
    cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;

    return cl;
}

f32 BackendCPU::coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& velocities, const View<f32, SingleSurface>& areas, const TensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) {    
    f32 cl = 0.0f;
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    const linalg::alias::float3 span_axis{0.f, 1.f, 0.f}; // TODO: obtain from the local frame
    const linalg::alias::float3 lift_axis = linalg::normalize(linalg::cross(freestream, span_axis));

    const i64 nc = gamma_delta.layout.nc();
    const i64 ns = gamma_delta.layout.ns();
    for (i64 i = 0; i < nc; i++) {
        for (i64 j = 0; j < ns; j++) {
            const i64 idx = i * gamma.layout.ld() + j; // linear index

            linalg::alias::float3 V{
                velocities[0*velocities.layout.stride() + idx],
                velocities[1*velocities.layout.stride() + idx],
                velocities[2*velocities.layout.stride() + idx]
            }; // local velocity (freestream + displacement vel)

            const i64 v0 = (i+0) * verts_wing.layout.ld() + j;
            const i64 v1 = (i+0) * verts_wing.layout.ld() + j + 1;

            const linalg::alias::float3 vertex0{verts_wing[0*verts_wing.layout.stride() + v0], verts_wing[1*verts_wing.layout.stride() + v0], verts_wing[2*verts_wing.layout.stride() + v0]}; // upper left
            const linalg::alias::float3 vertex1{verts_wing[0*verts_wing.layout.stride() + v1], verts_wing[1*verts_wing.layout.stride() + v1], verts_wing[2*verts_wing.layout.stride() + v1]}; // upper right
            const linalg::alias::float3 normal{normals(j, i, 0), normals(j, i, 1), normals(j, i, 2)};

            linalg::alias::float3 force = {0.0f, 0.0f, 0.0f};
            const f32 gamma_dt = (gamma[idx] - gamma_prev[idx]) / dt; // backward difference

            // Joukowski method
            force += rho * gamma_delta[idx] * linalg::cross(V, vertex1 - vertex0); // steady contribution
            force += rho * gamma_dt * areas[idx] * normal; // unsteady contribution

            // Katz Plotkin method
            // linalg::alias::float3 delta_p = {0.0f, 0.0f, 0.0f};
            // const f32 delta_gamma_i = (u == 0) ? gamma[li] : gamma[li] - gamma[(u-1) * mesh.ns + v];
            // const f32 delta_gamma_j = (v == 0) ? gamma[li] : gamma[li] - gamma[u * mesh.ns + v - 1];
            // delta_p += rho * linalg::dot(freestream, linalg::normalize(v1 - v0)) * delta_gamma_j / mesh.panel_width_y(u, v);
            // delta_p += rho * linalg::dot(freestream, linalg::normalize(v3 - v0)) * delta_gamma_i / mesh.panel_length(u, v);
            // delta_p += gamma_dt;
            // force = (delta_p * mesh.area[li]) * normal;

            // force /= linalg::length2(freestream);
            
            cl += linalg::dot(force, lift_axis);
        }
    }
    cl /= 0.5f * rho * linalg::length2(freestream) * area;

    return cl;
}

void BackendCPU::coeff_unsteady_cl_single_forces(
    const View<f32, SingleSurface>& verts_wing,
    const View<f32, SingleSurface>& gamma_delta,
    const View<f32, SingleSurface>& gamma,
    const View<f32, SingleSurface>& gamma_prev,
    const View<f32, SingleSurface>& velocities,
    const View<f32, SingleSurface>& areas,
    const TensorView3D<Location::Device>& normals,
    View<f32, SingleSurface>& forces,
    const linalg::alias::float3& freestream,
    f32 dt
    ) {
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    const i64 nc = gamma_delta.layout.nc();
    const i64 ns = gamma_delta.layout.ns();
    for (i64 i = 0; i < nc; i++) {
        for (i64 j = 0; j < ns; j++) {
            const i64 idx = i * gamma.layout.ld() + j; // linear index

            linalg::alias::float3 V{
                velocities[0*velocities.layout.stride() + idx],
                velocities[1*velocities.layout.stride() + idx],
                velocities[2*velocities.layout.stride() + idx]
            }; // local velocity (freestream + displacement vel)

            const i64 v0 = (i+0) * verts_wing.layout.ld() + j;
            const i64 v1 = (i+0) * verts_wing.layout.ld() + j + 1;

            const linalg::alias::float3 vertex0{verts_wing[0*verts_wing.layout.stride() + v0], verts_wing[1*verts_wing.layout.stride() + v0], verts_wing[2*verts_wing.layout.stride() + v0]}; // upper left
            const linalg::alias::float3 vertex1{verts_wing[0*verts_wing.layout.stride() + v1], verts_wing[1*verts_wing.layout.stride() + v1], verts_wing[2*verts_wing.layout.stride() + v1]}; // upper right
            const linalg::alias::float3 normal{normals(j, i, 0), normals(j, i, 1), normals(j, i, 2)};

            linalg::alias::float3 force = {0.0f, 0.0f, 0.0f};
            const f32 gamma_dt = (gamma[idx] - gamma_prev[idx]) / dt; // backward difference

            // Joukowski method
            force += rho * gamma_delta[idx] * linalg::cross(V, vertex1 - vertex0); // steady contribution
            force += rho * gamma_dt * areas[idx] * normal; // unsteady contribution

            forces[0*forces.layout.stride() + idx] = force.x;
            forces[1*forces.layout.stride() + idx] = force.y;
            forces[2*forces.layout.stride() + idx] = force.z;
        }
    }
}

// linalg::alias::float3 BackendCPU::coeff_steady_cm(
//     const FlowData& flow,
//     const f32 area,
//     const f32 mac,
//     const i64 j,
//     const i64 n)
// {
//     assert(n > 0);
//     assert(j >= 0 && j+n <= dd_mesh->ns);

//     const i64 nc = dd_mesh->nc;
//     const i64 ns = dd_mesh->ns;
//     const i64 nw = dd_mesh->nw;
//     const i64 nwa = dd_mesh->nwa;
//     const i64 nb_panels_wing = nc * ns;

//     linalg::alias::float3 cm(0.f, 0.f, 0.f);
//     const linalg::alias::float3 ref_pt{dd_mesh->frame + 12}; // frame origin as moment pt

//     for (i64 u = 0; u < nc; u++) {
//         for (i64 v = j; v < j + n; v++) {
//             const i64 li = u * ns + v; // linear index
//             const linalg::alias::float3 v0{*PTR_MESH_V(dd_mesh, u, v, 0), *PTR_MESH_V(dd_mesh, u, v, 1), *PTR_MESH_V(dd_mesh, u, v, 2)}; // upper left
//             const linalg::alias::float3 v1{*PTR_MESH_V(dd_mesh, u, v+1, 0), *PTR_MESH_V(dd_mesh, u, v+1, 1), *PTR_MESH_V(dd_mesh, u, v+1, 2)}; // upper right
//             // Leading edge vector pointing outward from wing root
//             const linalg::alias::float3 dl = v1 - v0;
//             // Distance from the center of leading edge to the reference point
//             const linalg::alias::float3 dst_to_ref = ref_pt - 0.5f * (v0 + v1);
//             // Distance from the center of leading edge to the reference point
//             const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * dd_data->delta_gamma[li];
//             cm += linalg::cross(force, dst_to_ref);
//         }
//     }
//     cm /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area * mac;
//     return cm;
// }

f32 BackendCPU::coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const View<f32, SingleSurface>& gamma_wake, const FlowData& flow, f32 area) {
    // tiny::ScopedTimer timer("Compute CD");
    f32 cd = ispc::kernel_trefftz_cd(verts_wake.ptr, verts_wake.layout.stride(), verts_wake.layout.nc(), verts_wake.layout.ns(), gamma_wake.ptr, sigma_vatistas);
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
void BackendCPU::mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& areas) {
    // parallel for
    for (int m = 0; m < colloc.size(); m++) {
        auto& colloc_i = colloc[m];
        auto& normals_i = normals[m];
        const f32* verts_wing_ptr = verts_wing.ptr + verts_wing.layout.offset(m);
        f32* areas_ptr = areas.ptr + areas.layout.offset(m);
        // parallel for
        for (i64 i = 0; i < colloc_i.shape(1); i++) {
            // inner vectorized loop
            for (i64 j = 0; j < colloc_i.shape(0); j++) {
                const i64 lidx = i * colloc_i.shape(0) + j;
                const i64 v0 = (i+0) * verts_wing.layout.ns(m) + j;
                const i64 v1 = (i+0) * verts_wing.layout.ns(m) + j + 1;
                const i64 v2 = (i+1) * verts_wing.layout.ns(m) + j + 1;
                const i64 v3 = (i+1) * verts_wing.layout.ns(m) + j;

                const linalg::alias::float3 vertex0{verts_wing_ptr[0*verts_wing.layout.stride() + v0], verts_wing_ptr[1*verts_wing.layout.stride() + v0], verts_wing_ptr[2*verts_wing.layout.stride() + v0]}; // upper left
                const linalg::alias::float3 vertex1{verts_wing_ptr[0*verts_wing.layout.stride() + v1], verts_wing_ptr[1*verts_wing.layout.stride() + v1], verts_wing_ptr[2*verts_wing.layout.stride() + v1]}; // upper right
                const linalg::alias::float3 vertex2{verts_wing_ptr[0*verts_wing.layout.stride() + v2], verts_wing_ptr[1*verts_wing.layout.stride() + v2], verts_wing_ptr[2*verts_wing.layout.stride() + v2]}; // lower right
                const linalg::alias::float3 vertex3{verts_wing_ptr[0*verts_wing.layout.stride() + v3], verts_wing_ptr[1*verts_wing.layout.stride() + v3], verts_wing_ptr[2*verts_wing.layout.stride() + v3]}; // lower left

                const linalg::alias::float3 normal_vec = linalg::normalize(linalg::cross(vertex3 - vertex1, vertex2 - vertex0));
                normals_i(j, i, 0) = normal_vec.x;
                normals_i(j, i, 1) = normal_vec.y;
                normals_i(j, i, 2) = normal_vec.z;

                // 3 vectors f (P0P3), b (P0P2), e (P0P1) to compute the area:
                // area = 0.5 * (||f x b|| + ||b x e||)
                // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
                const linalg::alias::float3 vec_f = vertex3 - vertex0;
                const linalg::alias::float3 vec_b = vertex2 - vertex0;
                const linalg::alias::float3 vec_e = vertex1 - vertex0;

                areas_ptr[lidx] = 0.5f * (linalg::length(linalg::cross(vec_f, vec_b)) + linalg::length(linalg::cross(vec_b, vec_e)));
                
                // High AoA correction (Aerodynamic Optimization of Aircraft Wings Using a Coupled VLM2.5D RANS Approach) Eq 3.4 p21
                // https://publications.polymtl.ca/2555/1/2017_MatthieuParenteau.pdf
                const f32 factor = (alpha_rad < EPS_f) ? 0.5f : 0.5f * (alpha_rad / (std::sin(alpha_rad) + EPS_f));
                const linalg::alias::float3 chord_vec = 0.5f * (vertex2 + vertex3 - vertex0 - vertex1);
                const linalg::alias::float3 colloc_pt = 0.5f * (vertex0 + vertex1) + factor * chord_vec;

                colloc_i(j, i, 0) = colloc_pt.x;
                colloc_i(j, i, 1) = colloc_pt.y;
                colloc_i(j, i, 2) = colloc_pt.z;
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
f32 BackendCPU::mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) {
    // Leading edge vertex
    f32* leading_edge_ptr = verts_wing.ptr;
    f32* trailing_edge_ptr = verts_wing.ptr + (verts_wing.layout.nc() - 1) * verts_wing.layout.ns();

    f32 mac = 0.0f;
    // loop over panel chordwise sections in spanwise direction
    // Note: can be done optimally with vertical fused simd
    for (i64 v = 0; v < areas.layout.ns(); v++) {
        // left and right chord lengths
        const f32 dx0 = trailing_edge_ptr[0*verts_wing.layout.stride() + v + 0] - leading_edge_ptr[0*verts_wing.layout.stride() + v + 0];
        const f32 dy0 = trailing_edge_ptr[1*verts_wing.layout.stride() + v + 0] - leading_edge_ptr[1*verts_wing.layout.stride() + v + 0];
        const f32 dz0 = trailing_edge_ptr[2*verts_wing.layout.stride() + v + 0] - leading_edge_ptr[2*verts_wing.layout.stride() + v + 0];
        const f32 dx1 = trailing_edge_ptr[0*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[0*verts_wing.layout.stride() + v + 1];
        const f32 dy1 = trailing_edge_ptr[1*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[1*verts_wing.layout.stride() + v + 1];
        const f32 dz1 = trailing_edge_ptr[2*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[2*verts_wing.layout.stride() + v + 1];
        const f32 c0 = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
        const f32 c1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
        // Panel width
        const f32 dx3 = leading_edge_ptr[0*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[0*verts_wing.layout.stride() + v + 0];
        const f32 dy3 = leading_edge_ptr[1*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[1*verts_wing.layout.stride() + v + 0];
        const f32 dz3 = leading_edge_ptr[2*verts_wing.layout.stride() + v + 1] - leading_edge_ptr[2*verts_wing.layout.stride() + v + 0];
        const f32 width = std::sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

        mac += 0.5f * (c0 * c0 + c1 * c1) * width;
    }
    // Since we divide by half the total wing area (both sides) we dont need to multiply by 2

    f32 wing_area = 0.0f;
    for (i64 i = 0; i < areas.layout.size(); i++) {
        wing_area += areas[i];
    }
    return mac / wing_area;
}

void BackendCPU::displace_wing(const TensorView<f32, 3, Location::Device>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) {
    // const tiny::ScopedTimer t("Mesh::move");
    assert(transforms.shape(2) == verts_wing.layout.surfaces().size());
    assert(verts_wing.layout.size() == verts_wing_init.layout.size());

    // TODO: parallel for
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        f32* vwing_ptr = verts_wing.ptr + verts_wing.layout.offset(i);
        f32* vwing_init_ptr = verts_wing_init.ptr + verts_wing_init.layout.offset(i);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 4, static_cast<i32>(verts_wing.layout.surface(i).size()), 4, 1.0f, transforms.ptr() + transforms.offset({0,0,i}), 4, vwing_init_ptr, static_cast<i32>(verts_wing_init.layout.stride()), 0.0f, vwing_ptr, static_cast<i32>(verts_wing.layout.stride()));
    }
}

void BackendCPU::wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, i32 iteration) {
    assert(verts_wing.layout.surfaces().size() == verts_wake.layout.surfaces().size());

    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        assert(iteration < verts_wake.layout.nc(i));
        f32* vwing = verts_wing.ptr + verts_wing.layout.offset(i) + (verts_wing.layout.nc(i) - 1) * verts_wing.layout.ns(i);
        f32* vwake = verts_wake.ptr + verts_wake.layout.offset(i) + (verts_wake.layout.nc(i) - iteration - 1) * verts_wake.layout.ns(i);

        memory->copy(Location::Device, vwake + 0*verts_wake.layout.stride(), 1, Location::Device, vwing + 0*verts_wing.layout.stride(), 1, sizeof(f32), verts_wing.layout.ns(i));
        memory->copy(Location::Device, vwake + 1*verts_wake.layout.stride(), 1, Location::Device, vwing + 1*verts_wing.layout.stride(), 1, sizeof(f32), verts_wing.layout.ns(i));
        memory->copy(Location::Device, vwake + 2*verts_wake.layout.stride(), 1, Location::Device, vwing + 2*verts_wing.layout.stride(), 1, sizeof(f32), verts_wing.layout.ns(i));
    }
}

// TODO: this is wrong, wont work if the surface is a slice of non uniform surface
// need to do a proper 2D loop and using the local ld
f32 BackendCPU::mesh_area(const View<f32, SingleSurface>& areas) {
    f32 wing_area = 0.0f;
    for (i64 i = 0; i < areas.layout.size(); i++) {
        wing_area += areas[i];
    }
    return wing_area;
}
// VLM
#include "vlm_backend_cuda.hpp"
#include "vlm_backend_cuda_kernels.cuh"
#include "vlm_executor.hpp"

// External
#include "tinytimer.hpp"
#include <taskflow/cuda/cudaflow.hpp>

// CUDA
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_math.cuh"
#include <vector_types.h>

// C++ STD
#include <cstdio>
#include <memory>

using namespace vlm;

#define CHECK_CUDA(call) \
    do { \
        const cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUSOLVER(call) \
    do { \
        const cusolverStatus_t err = (call); \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSolver Error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        const cublasStatus_t err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


void print_cuda_info() {
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    std::printf("DEVICE: %s (%d SMs, %llu MB, CUDA %d.%d)\n", device_props.name, device_props.multiProcessorCount, device_props.totalGlobalMem / (1024ull * 1024ull), device_props.major, device_props.minor);
}

// Singleton class to manage CUDA context (handle)
class CtxManager {
public:
    // This is how clients can access the single instance
    static CtxManager& getInstance() {
        static CtxManager instance;
        return instance;
    }

    void create() {
        // Create handlers
        CHECK_CUSOLVER(cusolverDnCreate(&m_cusolver));
        CHECK_CUBLAS(cublasCreate(&m_cublas));
        // Create CUDA stream
        CHECK_CUDA(cudaStreamCreate(&m_stream));
        // Set handler streams
        CHECK_CUSOLVER(cusolverDnSetStream(m_cusolver, m_stream));
        CHECK_CUBLAS(cublasSetStream(m_cublas, m_stream));
    }

    void destroy() {
        CHECK_CUSOLVER(cusolverDnDestroy(m_cusolver));
        CHECK_CUBLAS(cublasDestroy(m_cublas));
        CHECK_CUDA(cudaStreamDestroy(m_stream)); // destroy stream last
    }

    cusolverDnHandle_t cusolver() {return(m_cusolver);}
    cudaStream_t stream() {return(m_stream);}
    cublasHandle_t cublas() {return(m_cublas);}
    void sync() {CHECK_CUDA(cudaStreamSynchronize(m_stream));}

private:
    cusolverDnHandle_t m_cusolver = nullptr;
    cudaStream_t m_stream = nullptr;
    cublasHandle_t m_cublas = nullptr;
    
    CtxManager() = default;
    ~CtxManager() = default;
};

template<typename T>
void launch_fill_kernel(Location location, T* ptr, i64 stride, T value, i64 size) {
    switch (location) {
    case Location::Device: {
        constexpr Dim3<i32> block{1024};
        const Dim3<i64> n{(i64)size};
        kernel_fill<T><<<grid_size(block, n)(), block()>>>(ptr, stride, value, size);
        CHECK_CUDA(cudaGetLastError());
        break;
    }
    case Location::Host: {
        if (stride == 1) {
            std::fill(ptr, ptr + size, value);
        } else {
            for (i64 i = 0; i < size; i++) {
                ptr[i * stride] = value;
            }
        }
        break;
    }
    }
}

/// @brief Memory manager implementation for the CUDA backend
class CUDA_Memory final : public Memory {
    public:
        CUDA_Memory() : Memory(false) {}
        ~CUDA_Memory() = default;
        __host__ void* alloc(Location location, i64 size) const override {
            void* res = nullptr;
            switch (location) {
                case Location::Device: CHECK_CUDA(cudaMalloc(&res, size)); break;
                case Location::Host: CHECK_CUDA(cudaMallocHost(&res, size)); break;
            }
            return res;
        }

        __host__ void free(Location location, void* ptr) const override {
            switch (location) {
                case Location::Device: CHECK_CUDA(cudaFree(ptr)); break;
                case Location::Host: CHECK_CUDA(cudaFreeHost(ptr)); break;
            }
        }

        __host__ void copy(Location dst_loc, void* dst, i64 dst_stride, Location src_loc, const void* src, i64 src_stride, i64 elem_size, i64 size) const override {
            cudaMemcpyKind kind;
            if (dst_loc == Location::Host && src_loc == Location::Device) {
                kind = cudaMemcpyDeviceToHost;
            } else if (dst_loc == Location::Device && src_loc == Location::Host) {
                kind = cudaMemcpyHostToDevice;
            } else if (dst_loc == Location::Device && src_loc == Location::Device) {
                kind = cudaMemcpyDeviceToDevice;
            } else if (dst_loc == Location::Host && src_loc == Location::Host) {
                kind = cudaMemcpyHostToHost;
            }

            if (dst_stride == 1 && src_stride == 1) {
                CHECK_CUDA(cudaMemcpy(dst, src, size * elem_size, kind));
            } else {
                CHECK_CUDA(cudaMemcpy2D(dst, dst_stride * elem_size, src, src_stride * elem_size, elem_size, size, kind));
            }
        }

        __host__ void fill(Location location, float* ptr, i64 stride, float value, i64 size) const override { launch_fill_kernel(location, ptr, stride, value, size); }
        __host__ void fill(Location location, double *ptr, i64 stride, double value, i64 size) const override { launch_fill_kernel(location, ptr, stride, value, size); }
};

BackendCUDA::BackendCUDA() : Backend(std::make_unique<CUDA_Memory>())  {
    print_cuda_info();
    CtxManager::getInstance().create();
}

BackendCUDA::~BackendCUDA() {
    CtxManager::getInstance().destroy();
}

std::unique_ptr<Memory> BackendCUDA::create_memory_manager() { return std::make_unique<CUDA_Memory>(); }
// std::unique_ptr<Kernels> create_kernels() { return std::make_unique<CPU_Kernels>(); }

void BackendCUDA::gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) {
    assert(gamma_delta.layout.dims() == 1);
    assert(gamma.layout.dims() == 1);
    
    for (const auto& surf : gamma_delta.layout.surfaces())  {
        f32* s_gamma_delta = gamma_delta.ptr + surf.offset;
        const f32* s_gamma = gamma.ptr + surf.offset;
        memory->copy(Location::Device, s_gamma_delta, 1, Location::Device, s_gamma, 1, sizeof(*s_gamma_delta), surf.ns);
        
        constexpr Dim3<i32> block{32, 32};
        const Dim3<i64> n{surf.nc-1, surf.ns};
        kernel_gamma_delta<block.x, block.y><<<grid_size(block, n)(), block()>>>(n.x, n.y, s_gamma_delta + surf.ns, s_gamma + surf.ns);
        CHECK_CUDA(cudaGetLastError());
    };

    // tf::Taskflow taskflow;

    // auto kernels = taskflow.emplace([&](tf::cudaFlow& cf){
    //     for (const auto& surf : gamma_delta.layout.surfaces())  {
    //         f32* s_gamma_delta = gamma_delta.ptr + surf.offset;
    //         const f32* s_gamma = gamma.ptr + surf.offset;
    //         tf::cudaTask copy = cf.copy(s_gamma_delta, s_gamma, surf.ns * sizeof(*s_gamma_delta));
                
    //         constexpr Dim3<i32> block{32, 32};
    //         const Dim3<i64> n{surf.nc-1, surf.ns};
    //         tf::cudaTask kernel = cf.kernel(grid_size(block, n), block, 0, kernel_gamma_delta, n.x, n.y, s_gamma_delta + surf.ns, s_gamma + surf.ns).succede(copy);
    //     };
    // });
    // auto check = taskflow.emplace([&](){
    //     CHECK_CUDA(cudaGetLastError());
    // }).succede(kernels);

    // Executor::get().run(taskflow).wait();
}

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
void BackendCUDA::lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<i32>& condition, i32 iteration) {
    assert(condition.size() == colloc.layout.surfaces().size());
    std::fill(condition.begin(), condition.end(), 0); // reset conditon increment vars

    for (i32 i = 0; i < colloc.layout.surfaces().size(); i++) {
        f32* lhs_section = lhs.ptr() + colloc.layout.offset(i) * lhs.stride(1);
        
        f32* vwing_section = verts_wing.ptr + verts_wing.layout.offset(i);
        f32* vwake_section = verts_wake.ptr + verts_wake.layout.offset(i);
        const i64 end_wing = (colloc.layout.nc(i) - 1) * colloc.layout.ns(i);
        
        constexpr Dim3<i32> block{32, 16};
        const Dim3<i64> n{lhs.shape(0), end_wing};
        // wing pass
        kernel_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
            n.x,
            n.y,
            lhs_section,
            colloc.ptr,
            colloc.layout.stride(),
            vwing_section,
            verts_wing.layout.stride(),
            verts_wing.layout.ns(i),
            normals.ptr, normals.layout.stride(),
            sigma_vatistas
        );
        CHECK_CUDA(cudaGetLastError());

        const Dim3<i64> n2{lhs.shape(0), colloc.layout.ns(i)};
        // last wing row
        kernel_influence<block.x, block.y><<<grid_size(block, n2)(), block()>>>(
            n2.x,
            n2.y, 
            lhs_section + end_wing * lhs.stride(1),
            colloc.ptr,
            colloc.layout.stride(),
            vwing_section + (verts_wing.layout.nc(i)-2)*verts_wing.layout.ns(i),
            verts_wing.layout.stride(),
            verts_wing.layout.ns(i),
            normals.ptr,
            normals.layout.stride(),
            sigma_vatistas
        );
        CHECK_CUDA(cudaGetLastError());

        while (condition[i] < iteration) {
            const Dim3<i64> n3{lhs.shape(0), colloc.layout.ns(i)};
            // each wake row
            kernel_influence<block.x, block.y><<<grid_size(block, n3)(), block()>>>(
                n3.x,
                n3.y,
                lhs_section + end_wing * lhs.stride(1),
                colloc.ptr,
                colloc.layout.stride(),
                vwake_section + (verts_wake.layout.nc(i) - condition[i] - 2) * verts_wake.layout.ns(i),
                verts_wake.layout.stride(),
                verts_wake.layout.ns(i), 
                normals.ptr,
                normals.layout.stride(),
                sigma_vatistas
            );
            CHECK_CUDA(cudaGetLastError());
            condition[i]++;
        }
    }
}

/// @brief Add velocity contributions to the right hand side vector
/// @details
/// Add velocity contributions to the right hand side vector of the VLM system
/// @param rhs right hand side vector
/// @param normals normals of all surfaces
/// @param velocities displacement velocities of all surfaces
void BackendCUDA::rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& velocities) {
    // const tiny::ScopedTimer timer("RHS");

    constexpr Dim3<i32> block{768};
    const Dim3<i64> n{rhs.size()};
    kernel_rhs_assemble_velocities<block.x><<<grid_size(block, n)(), block()>>>(n.x, rhs.ptr(), velocities.ptr, velocities.layout.stride(), normals.ptr, normals.layout.stride());
}

void BackendCUDA::rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const View<f32, MultiSurface>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const View<f32, MultiSurface>& verts_wake, i32 iteration) {
    if (iteration == 0) return; // cuda doesnt support 0 sized domain
    constexpr Dim3<i32> block{32, 16};

    for (i32 i = 0; i < normals.layout.surfaces().size(); i++) {
        const i64 wake_m  = iteration;
        const i64 wake_n  = verts_wake.layout.ns(i) - 1;
        const Dim3<i64> n{wake_m * wake_n, rhs.size()};
        kernel_wake_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
            wake_m, 
            wake_n,
            rhs.size(),
            colloc.ptr,
            colloc.layout.stride(),
            normals.ptr,
            normals.layout.stride(),
            verts_wake.ptr + verts_wake.layout.offset(i) + (verts_wake.layout.nc(i) - iteration - 1) * verts_wake.layout.ns(i),
            verts_wake.layout.stride(),
            gamma_wake.ptr + gamma_wake.layout.offset(i) + (gamma_wake.layout.nc(i) - iteration) * gamma_wake.layout.ns(i),
            rhs.ptr(),
            sigma_vatistas
        );
        CHECK_CUDA(cudaGetLastError());
    }
}

void BackendCUDA::displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, i32 iteration) {
    // TODO
}

void BackendCUDA::gamma_shed(View<f32, MultiSurface>& gamma_wing, View<f32, MultiSurface>& gamma_wing_prev, View<f32, MultiSurface>& gamma_wake, i32 iteration) {
    // const tiny::ScopedTimer timer("Shed Gamma");

    memory->copy(Location::Device, gamma_wing_prev.ptr, 1, Location::Device, gamma_wing.ptr, 1, sizeof(f32), gamma_wing.size());
    for (i64 i = 0; i < gamma_wake.layout.surfaces().size(); i++) {
        assert(iteration < gamma_wake.layout.nc(i));
        f32* gamma_wake_ptr = gamma_wake.ptr + gamma_wake.layout.offset(i) + (gamma_wake.layout.nc(i) - iteration - 1) * gamma_wake.layout.ns(i);
        f32* gamma_wing_ptr = gamma_wing.ptr + gamma_wing.layout.offset(i) + (gamma_wing.layout.nc(i) - 1) * gamma_wing.layout.ns(i); // last row
        memory->copy(Location::Device, gamma_wake_ptr, 1, Location::Device, gamma_wing_ptr, 1, sizeof(f32), gamma_wing.layout.ns(i));
    }
}

f32 BackendCUDA::coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow, f32 area) {
    f32 h_cl = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{gamma_delta.layout.ns(), gamma_delta.layout.nc()};
    kernel_coeff_steady_cl_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        n.y,
        n.x,
        verts_wing.ptr,
        verts_wing.layout.ld(),
        verts_wing.layout.stride(),
        gamma_delta.ptr,
        gamma_delta.layout.ld(),
        float3{flow.freestream.x, flow.freestream.y, flow.freestream.z},
        float3{flow.lift_axis.x, flow.lift_axis.y, flow.lift_axis.z},
        flow.rho,
        d_val
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_cl, 1, Location::Device, d_val, 1, sizeof(f32), 1);
    CtxManager::getInstance().sync();
    
    return h_cl / (0.5f * flow.rho * linalg::length2(flow.freestream) * area);
}

f32 BackendCUDA::coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const auto gamma_delta_local = gamma_delta.layout.subview(gamma_delta.ptr, i);
        const auto areas_local = areas.layout.subview(areas.ptr, i);
        const f32 area_local = mesh_area(areas_local);
        const f32 wing_cl = coeff_steady_cl_single(verts_wing_local, gamma_delta_local, flow, area_local);
        cl += wing_cl * area_local;
        total_area += area_local;
    }
    cl /= total_area;
    return cl;
}

f32 BackendCUDA::coeff_unsteady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) {    
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    f32 h_cl = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{gamma.layout.ns(), gamma.layout.nc()};
    kernel_coeff_unsteady_cl_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        n.y,
        n.x,
        gamma.layout.ld(),
        verts_wing.ptr,
        verts_wing.layout.stride(),
        gamma_delta.ptr,
        gamma.ptr,
        gamma_prev.ptr,
        velocities.ptr,
        velocities.layout.stride(),
        areas.ptr,
        normals.ptr,
        normals.layout.stride(),
        float3{freestream.x, freestream.y, freestream.z},
        dt,
        d_val
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_cl, 1, Location::Device, d_val, 1, sizeof(f32), 1);
    CtxManager::getInstance().sync();
    
    return h_cl / (0.5f * rho * linalg::length2(freestream) * area);
}

f32 BackendCUDA::coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const MultiTensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt) {
        f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        const auto verts_wing_local = verts_wing.layout.subview(verts_wing.ptr, i);
        const auto areas_local = areas.layout.subview(areas.ptr, i);
        const auto gamma_delta_local = gamma_wing_delta.layout.subview(gamma_wing_delta.ptr, i);
        const auto gamma_wing_local = gamma_wing.layout.subview(gamma_wing.ptr, i);
        const auto gamma_wing_prev_local = gamma_wing_prev.layout.subview(gamma_wing_prev.ptr, i);
        const auto velocities_local = velocities.layout.subview(velocities.ptr, i);
        const auto normals_local = normals.layout.subview(normals.ptr, i);

        const f32 area_local = mesh_area(areas_local);
        const f32 wing_cl = coeff_unsteady_cl_single(verts_wing_local, gamma_delta_local, gamma_wing_local, gamma_wing_prev_local, velocities_local, areas_local, normals_local, freestream, dt, area_local);
        cl += wing_cl * area_local;
        total_area += area_local;
    }
    cl /= total_area;
    return cl;
}

void BackendCUDA::coeff_unsteady_cl_single_forces(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const View<f32, SingleSurface>& gamma, const View<f32, SingleSurface>& gamma_prev, const View<f32, SingleSurface>& velocities, const View<f32, SingleSurface>& areas, const View<f32, SingleSurface>& normals, View<f32, SingleSurface>& forces, const linalg::alias::float3& freestream, f32 dt) {}

void BackendCUDA::coeff_unsteady_cl_multi_forces(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& forces, const linalg::alias::float3& freestream, f32 dt) {}

f32 BackendCUDA::coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const View<f32, SingleSurface>& gamma_wake, const FlowData& flow, f32 area) {
    f32 h_cd = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<i32> block{64, 8};
    const Dim3<i64> n{verts_wake.layout.ns()-1, verts_wake.layout.ns()-1};
    kernel_coeff_steady_cd_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        verts_wake.ptr,
        verts_wake.layout.stride(),
        verts_wake.layout.nc(),
        verts_wake.layout.ns(),
        gamma_wake.ptr,
        sigma_vatistas,
        d_val
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_cd, 1, Location::Device, d_val, 1, sizeof(f32), 1);
    CtxManager::getInstance().sync();
    return h_cd / (linalg::length2(flow.freestream) * area);
}

// TODO: move in backend.cpp ?
f32 BackendCUDA::coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& gamma_wake, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cd = 0.0f;
    f32 total_area = 0.0f;
    for (i64 i = 0; i < verts_wake.layout.surfaces().size(); i++) {
        const auto verts_wake_local = verts_wake.layout.subview(verts_wake.ptr, i);
        const auto gamma_wake_local = gamma_wake.layout.subview(gamma_wake.ptr, i);
        const auto areas_local = areas.layout.subview(areas.ptr, i);
        const f32 area_local = mesh_area(areas_local);
        const f32 wing_cd = coeff_steady_cd_single(verts_wake_local, gamma_wake_local, flow, area_local);
        cd += wing_cd * area_local;
        total_area += area_local;
    }
    cd /= total_area;
    return cd;
}

void BackendCUDA::mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, View<f32, MultiSurface>& areas) {
    for (int m = 0; m < colloc.layout.surfaces().size(); m++) {
        const f32* verts_wing_ptr = verts_wing.ptr + verts_wing.layout.offset(m);
        f32* colloc_ptr = colloc.ptr + colloc.layout.offset(m);
        f32* normals_ptr = normals.ptr + normals.layout.offset(m);
        f32* areas_ptr = areas.ptr + areas.layout.offset(m);

        constexpr Dim3<i32> block{24, 32}; // ns, nc
        const Dim3<i64> n{colloc.layout.ns(m), colloc.layout.nc(m)};
        kernel_mesh_metrics<block.x, block.y><<<grid_size(block, n)(), block()>>>(n.y, n.x, colloc_ptr, colloc.layout.stride(), normals_ptr, normals.layout.stride(), areas_ptr, verts_wing_ptr, verts_wing.layout.stride(), alpha_rad);
    }
}

f32 BackendCUDA::mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) {
    // TODO
    return 0.0f;
}

void BackendCUDA::displace_wing(const TensorView<f32, 3, Location::Device>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) {
    // const tiny::ScopedTimer t("Mesh::move");
    assert(transforms.shape(2) == verts_wing.layout.surfaces().size());
    assert(verts_wing.layout.size() == verts_wing_init.layout.size());

    const f32 alpha = 1.0f;
    const f32 beta = 0.0f;

    // TODO: parallel for
    for (i64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        CHECK_CUBLAS(cublasSgemm(
            CtxManager::getInstance().cublas(),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            static_cast<int>(verts_wing.layout.surface(i).size()),
            4,            
            4,
            &alpha,
            verts_wing_init.ptr + verts_wing_init.layout.offset(i),
            static_cast<int>(verts_wing_init.layout.stride()),
            transforms.ptr() + transforms.offset({0,0,i}),
            4,
            &beta,
            verts_wing.ptr + verts_wing.layout.offset(i),
            static_cast<int>(verts_wing.layout.stride())
        ));
    }
}

void BackendCUDA::wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, i32 iteration) {
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

f32 BackendCUDA::mesh_area(const View<f32, SingleSurface>& areas) {
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<i32> block{512}; // ns, nc
    const Dim3<i64> n{static_cast<i64>(areas.layout.size())};
    kernel_reduce<<<grid_size(block, n)(), block()>>>(n.x, areas.ptr, d_val);
    CHECK_CUDA(cudaGetLastError());
    f32 h_area;
    memory->copy(Location::Host, &h_area, 1, Location::Device, d_val, 1, sizeof(f32), 1);
    return h_area;
}
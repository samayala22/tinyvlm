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

BackendCUDA::BackendCUDA() : Backend(create_memory_manager(), create_blas())  {
    print_cuda_info();
    CtxManager::getInstance().create();
}

BackendCUDA::~BackendCUDA() {
    CtxManager::getInstance().destroy();
}

std::unique_ptr<Memory> BackendCUDA::create_memory_manager() { return std::make_unique<CUDA_Memory>(); }
// std::unique_ptr<Kernels> create_kernels() { return std::make_unique<CPU_Kernels>(); }

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
void BackendCUDA::lhs_assemble(TensorView<f32, 2, Location::Device>& lhs, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView3D<Location::Device>& verts_wake, std::vector<i32>& condition, i32 iteration) {
    // tiny::ScopedTimer timer("LHS");
    std::fill(condition.begin(), condition.end(), 0); // reset conditon increment vars
    constexpr Dim3<i32> block{32, 16};
    tf::Taskflow graph;

    auto begin = graph.placeholder();
    auto end = graph.placeholder();

    i64 offset_j = 0;
    for (i32 m_j = 0; m_j < colloc.size(); m_j++) {
        i64 offset_i = 0;
        for (i32 m_i = 0; m_i < colloc.size(); m_i++) {
            const i64 condition_idx = m_i + m_j * static_cast<i64>(colloc.size());
            auto colloc_i = colloc[m_i];
            auto colloc_j = colloc[m_j];
            auto normals_i = normals[m_i];

            f32* lhs_section = lhs.ptr() + offset_i + offset_j * lhs.stride(1);
            f32* vwing_section = verts_wing.ptr + verts_wing.layout.offset(m_j);
            f32* vwake_section = verts_wake.ptr + verts_wake.layout.offset(m_j);

            const i64 zero = 0;
            const i64 end_wing = (colloc_j.shape(1) - 1) * colloc_j.shape(0);
            
            auto wing_pass = graph.emplace([=](){
                const Dim3<i64> n{colloc_i.stride(2), end_wing};
                kernel_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
                    n.x,
                    n.y,
                    lhs_section,
                    lhs.stride(1),
                    colloc_i.ptr(),
                    colloc_i.stride(2),
                    vwing_section,
                    verts_wing.layout.stride(),
                    verts_wing.layout.ns(m_j),
                    normals_i.ptr(),
                    normals_i.stride(2),
                    sigma_vatistas
                );
                CHECK_CUDA(cudaGetLastError());
            }).name("wing pass");

            auto last_row = graph.emplace([=](){
                const Dim3<i64> n{colloc_i.stride(2), colloc_j.shape(0)};
                kernel_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
                    n.x,
                    n.y,
                    lhs_section + end_wing * lhs.stride(1),
                    lhs.stride(1),
                    colloc_i.ptr(),
                    colloc_i.stride(2),
                    vwing_section + (verts_wing.layout.nc(m_j)-2)*verts_wing.layout.ns(m_j),
                    verts_wing.layout.stride(),
                    verts_wing.layout.ns(m_j),
                    normals_i.ptr(),
                    normals_i.stride(2),
                    sigma_vatistas
                );
                CHECK_CUDA(cudaGetLastError());
            }).name("last_row");            

            auto cond = graph.emplace([=, &condition] {
                return condition[condition_idx] < iteration ? 0 : 1; // 0 means continue, 1 means break (exit loop)
            }).name("condition");

            auto wake_pass = graph.emplace([=, &condition](){
                const Dim3<i64> n{colloc_i.stride(2), colloc_j.shape(0)};
                kernel_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
                    n.x,
                    n.y,
                    lhs_section + end_wing * lhs.stride(1),
                    lhs.stride(1),
                    colloc_i.ptr(),
                    colloc_i.stride(2),
                    vwake_section + (verts_wake.layout.nc(m_j) - condition[condition_idx] - 2) * verts_wake.layout.ns(m_j),
                    verts_wake.layout.stride(),
                    verts_wake.layout.ns(m_j), 
                    normals_i.ptr(),
                    normals_i.stride(2),
                    sigma_vatistas
                );
                CHECK_CUDA(cudaGetLastError());
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
void BackendCUDA::rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& velocities) {
    // const tiny::ScopedTimer timer("RHS");
    
    i64 offset = 0;
    for (i64 m = 0; m < normals.size(); m++) {
        assert(offset <= rhs.size());
        const auto& normals_i = normals[m];
        constexpr Dim3<i32> block{768};
        const Dim3<i64> n{normals_i.stride(2)};
        kernel_rhs_assemble_velocities<block.x><<<grid_size(block, n)(), block()>>>(
            n.x,
            rhs.ptr() + offset,
            velocities.ptr,
            velocities.layout.stride(),
            normals_i.ptr(),
            normals_i.stride(2)
        );
        offset += normals_i.stride(2);
    }
}

void BackendCUDA::rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView2D<Location::Device>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& verts_wake, i32 iteration) {
    if (iteration == 0) return; // cuda doesnt support 0 sized domain
    constexpr Dim3<i32> block{32, 16};

    for (i32 i = 0; i < normals.size(); i++) {
        const auto& colloc_i = colloc[i];
        const auto& normals_i = normals[i];
        const auto& gamma_wake_i = gamma_wake[i];
        const i64 wake_m  = iteration;
        const i64 wake_n  = verts_wake.layout.ns(i) - 1;
        const Dim3<i64> n{wake_m * wake_n, rhs.size()};
        kernel_wake_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
            wake_m, 
            wake_n,
            rhs.size(),
            colloc_i.ptr(),
            colloc_i.stride(2),
            normals_i.ptr(),
            normals_i.stride(2),
            verts_wake.ptr + verts_wake.layout.offset(i) + (verts_wake.layout.nc(i) - iteration - 1) * verts_wake.layout.ns(i),
            verts_wake.layout.stride(),
            gamma_wake_i.ptr() + gamma_wake_i.offset({0, gamma_wake_i.shape(1) - iteration}),
            rhs.ptr(),
            sigma_vatistas
        );
        CHECK_CUDA(cudaGetLastError());
    }
}

void BackendCUDA::displace_wake_rollup(MultiTensorView3D<Location::Device>& wake_rollup, const MultiTensorView3D<Location::Device>& verts_wake, const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wake, f32 dt, i32 iteration) {
    // TODO
}

f32 BackendCUDA::coeff_steady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const FlowData& flow, f32 area) {
    f32 h_cl = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{gamma_delta.shape(0), gamma_delta.shape(1)};
    kernel_coeff_steady_cl_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        n.y,
        n.x,
        verts_wing.ptr,
        verts_wing.layout.ld(),
        verts_wing.layout.stride(),
        gamma_delta.ptr(),
        gamma_delta.stride(1),
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

f32 BackendCUDA::coeff_unsteady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const TensorView2D<Location::Device>& gamma, const TensorView2D<Location::Device>& gamma_prev, const TensorView3D<Location::Device>& velocities, const TensorView2D<Location::Device>& areas, const TensorView3D<Location::Device>& normals, const linalg::alias::float3& freestream, f32 dt, f32 area) {    
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    f32 h_cl = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{gamma_delta.shape(0), gamma_delta.shape(1)};
    kernel_coeff_unsteady_cl_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        n.y,
        n.x,
        gamma.stride(1),
        verts_wing.ptr,
        verts_wing.layout.stride(),
        gamma_delta.ptr(),
        gamma.ptr(),
        gamma_prev.ptr(),
        velocities.ptr,
        velocities.layout.stride(),
        areas.ptr(),
        normals.ptr(),
        normals.stride(2),
        float3{freestream.x, freestream.y, freestream.z},
        dt,
        d_val
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_cl, 1, Location::Device, d_val, 1, sizeof(f32), 1);
    CtxManager::getInstance().sync();
    
    return h_cl / (0.5f * rho * linalg::length2(freestream) * area);
}

void BackendCUDA::coeff_unsteady_cl_single_forces(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const TensorView2D<Location::Device>& gamma, const TensorView2D<Location::Device>& gamma_prev, const TensorView3D<Location::Device>& velocities, const TensorView2D<Location::Device>& areas, const TensorView3D<Location::Device>& normals, TensorView3D<Location::Device>& forces, const linalg::alias::float3& freestream, f32 dt) {}

f32 BackendCUDA::coeff_steady_cd_single(const TensorView3D<Location::Device>& verts_wake, const TensorView2D<Location::Device>& gamma_wake, const FlowData& flow, f32 area) {
    f32 h_cd = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<i32> block{64, 8};
    const Dim3<i64> n{verts_wake.layout.ns()-1, verts_wake.layout.ns()-1};
    kernel_coeff_steady_cd_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        verts_wake.ptr,
        verts_wake.layout.stride(),
        verts_wake.layout.nc(),
        verts_wake.layout.ns(),
        gamma_wake.ptr(),
        sigma_vatistas,
        d_val
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_cd, 1, Location::Device, d_val, 1, sizeof(f32), 1);
    CtxManager::getInstance().sync();
    return h_cd / (linalg::length2(flow.freestream) * area);
}

void BackendCUDA::mesh_metrics(const f32 alpha_rad, const MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, MultiTensorView2D<Location::Device>& areas) {
    for (int m = 0; m < colloc.size(); m++) {
        const auto& colloc_i = colloc[m];
        const auto& normals_i = normals[m];
        const auto& areas_i = areas[m];
        const f32* verts_wing_ptr = verts_wing.ptr + verts_wing.layout.offset(m);

        constexpr Dim3<i32> block{24, 32}; // ns, nc
        const Dim3<i64> n{colloc_i.shape(0), colloc_i.shape(1)};
        kernel_mesh_metrics<block.x, block.y><<<grid_size(block, n)(), block()>>>(
            n.y,
            n.x,
            colloc_i.ptr(),
            colloc_i.stride(2),
            normals_i.ptr(),
            normals_i.stride(2),
            areas_i.ptr(),
            verts_wing_ptr,
            verts_wing.layout.stride(),
            alpha_rad
        );
    }
}

f32 BackendCUDA::mesh_mac(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& areas) {
    // TODO
    return 0.0f;
}

void BackendCUDA::displace_wing(const TensorView<f32, 3, Location::Device>& transforms, MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& verts_wing_init) {
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

void BackendCUDA::wake_shed(const MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& verts_wake, i32 iteration) {
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

f32 BackendCUDA::sum(const TensorView1D<Location::Device>& tensor) {
    f32* d_sum = (f32*)memory->alloc(Location::Device, sizeof(*d_sum));
    f32 h_sum;
    cudaMemset(d_sum, 0, sizeof(f32)); // TODO: use fill instead ?
    constexpr Dim3<i32> block{768};
    const Dim3<i64> n{tensor.size()};
    kernel_reduce<<<grid_size(block, n)(), block()>>>(
        tensor.shape(0),
        tensor.ptr(),
        d_sum
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_sum, 1, Location::Device, d_sum, 1, sizeof(f32), 1);
    memory->free(Location::Device, d_sum);
    return h_sum;
}

f32 BackendCUDA::sum(const TensorView2D<Location::Device>& tensor) {
    f32* d_sum = (f32*)memory->alloc(Location::Device, sizeof(*d_sum));
    f32 h_sum;
    cudaMemset(d_sum, 0, sizeof(f32));
    constexpr Dim3<i32> block{768};
    const Dim3<i64> n{tensor.size()};
    kernel_reduce_2D<<<grid_size(block, n)(), block()>>>(
        tensor.shape(0),
        tensor.shape(1),
        tensor.ptr(),
        tensor.stride(1),
        d_sum
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_sum, 1, Location::Device, d_sum, 1, sizeof(f32), 1);
    memory->free(Location::Device, d_sum);
    return h_sum;
}
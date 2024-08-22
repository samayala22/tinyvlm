#include "vlm_backend_cuda.hpp"
#include "vlm_backend_cuda_kernels.cuh"
#include "vlm_executor.hpp"

#include "tinytimer.hpp"

#include <taskflow/cuda/cudaflow.hpp>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_math.cuh"

#include <cstdio>
#include <stdlib.h>
#include <vector_types.h>

using namespace vlm;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t err = (call); \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSolver Error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    std::printf("----- CUDA Device information -----\n");
    std::printf("Found %d device(s)\n", deviceCount);
    // Get CUDA Runtime version
    int cudaRuntimeVersion = 0;
    cudaRuntimeGetVersion(&cudaRuntimeVersion);
    std::printf("CUDA Runtime: %d.%d\n", cudaRuntimeVersion / 1000, (cudaRuntimeVersion % 100) / 10);
    
    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        std::printf("Device %d: %s\n", i, deviceProps.name);
        std::printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        std::printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        std::printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    std::printf("-----------------------------------\n");
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

private:
    cusolverDnHandle_t m_cusolver = nullptr;
    cudaStream_t m_stream = nullptr;
    cublasHandle_t m_cublas = nullptr;
    
    CtxManager() = default;
    ~CtxManager() = default;
};

/// @brief Memory manager implementation for the CPU backend
class MemoryCUDA final : public Memory {
    public:
        MemoryCUDA() : Memory(false) {}
        ~MemoryCUDA() = default;
        void* alloc(MemoryLocation location, std::size_t size) const override {
            void* res = nullptr;
            switch (location) {
                case MemoryLocation::Device: CHECK_CUDA(cudaMalloc(&res, size)); break;
                case MemoryLocation::Host: CHECK_CUDA(cudaMallocHost(&res, size)); break;
            }
            return res;
        }

        void free(MemoryLocation location, void* ptr) const override {
            switch (location) {
                case MemoryLocation::Device: CHECK_CUDA(cudaFree(ptr)); break;
                case MemoryLocation::Host: CHECK_CUDA(cudaFreeHost(ptr)); break;
            }
        }

        void copy(MemoryTransfer transfer, void* dst, const void* src, std::size_t size) const override {
            switch (transfer) {
                case MemoryTransfer::HostToDevice: CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)); break;
                case MemoryTransfer::DeviceToHost: CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)); break;
                case MemoryTransfer::HostToHost: CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyHostToHost)); break;
                case MemoryTransfer::DeviceToDevice: CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)); break;
            }
        }
        void fill_f32(MemoryLocation location, float* ptr, float value, std::size_t size) const override {
            switch (location) {
                case MemoryLocation::Device: {
                    constexpr Dim3<u32> block{1024};
                    const Dim3<u64> n{size};
                    kernel_fill_f32<block><<<grid_size(block, n)(), block()>>>(ptr, value, size);
                    CHECK_CUDA(cudaGetLastError());
                    break;
                }
                case MemoryLocation::Host: {
                    std::fill(ptr, ptr + size, value);
                    break;
                }
            }
        };
};

BackendCUDA::BackendCUDA() : Backend(std::make_unique<MemoryCUDA>())  {
    printCudaInfo();
    CtxManager::getInstance().create();

    // Prepare LU solver buffers
    // int bufsize = 0;
    // CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(ctx.cusolver(), n, n, d_lhs, n, &bufsize));
    // CHECK_CUDA(cudaMalloc((void**)&d_solver_info, sizeof(int)));
    // CHECK_CUDA(cudaMalloc((void**)&d_solver_buffer, sizeof(float) * bufsize));
    // CHECK_CUDA(cudaMalloc((void**)&d_solver_ipiv, sizeof(int) * n));
}

BackendCUDA::~BackendCUDA() {
    CtxManager::getInstance().destroy();
}

void BackendCUDA::gamma_delta(View<f32, MultiSurface>& gamma_delta, const View<f32, MultiSurface>& gamma) {
    assert(gamma_delta.layout.dims() == 1);
    assert(gamma.layout.dims() == 1);
    
    for (const auto& surf : gamma_delta.layout.surfaces())  {
        f32* s_gamma_delta = gamma_delta.ptr + surf.offset;
        const f32* s_gamma = gamma.ptr + surf.offset;
        memory->copy(MemoryTransfer::DeviceToDevice, s_gamma_delta, s_gamma, surf.ns * sizeof(*s_gamma_delta));
        
        constexpr Dim3<u32> block{32, 32};
        const Dim3<u64> n{surf.nc-1, surf.ns};
        kernel_gamma_delta<block><<<grid_size(block, n)(), block()>>>(n.x, n.y, s_gamma_delta + surf.ns, s_gamma + surf.ns);
        CHECK_CUDA(cudaGetLastError());
    };

    // tf::Taskflow taskflow;

    // auto kernels = taskflow.emplace([&](tf::cudaFlow& cf){
    //     for (const auto& surf : gamma_delta.layout.surfaces())  {
    //         f32* s_gamma_delta = gamma_delta.ptr + surf.offset;
    //         const f32* s_gamma = gamma.ptr + surf.offset;
    //         tf::cudaTask copy = cf.copy(s_gamma_delta, s_gamma, surf.ns * sizeof(*s_gamma_delta));
                
    //         constexpr Dim3<u32> block{32, 32};
    //         const Dim3<u64> n{surf.nc-1, surf.ns};
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
void BackendCPU::lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<u32>& condition, u32 iteration) {    tiny::ScopedTimer timer("LHS");
    assert(condition.size() == colloc.layout.surfaces().size());
    std::fill(condition.begin(), condition.end(), 0); // reset conditon increment vars

    for (u32 i = 0; i < colloc.layout.surfaces().size(); i++) {
        f32* lhs_section = lhs.ptr + colloc.layout.offset(i) * lhs.layout.stride();
        
        f32* vwing_section = verts_wing.ptr + verts_wing.layout.offset(i);
        f32* vwake_section = verts_wake.ptr + verts_wake.layout.offset(i);
        const u64 zero = 0;
        const u64 end_wing = (colloc.layout.nc(i) - 1) * colloc.layout.ns(i);
        
        constexpr Dim3<u32> block{32, 16};
        const Dim3<u64> n{lhs.layout.m(), end_wing};
        // wing pass
        kernel_influence<block><<<grid_size(block, n)(), block()>>>(n.x, n.y, lhs_section, colloc.ptr, colloc.layout.stride(), vwing_section, verts_wing.layout.stride(), verts_wing.layout.ns(i), normals.ptr, normals.layout.stride(), sigma_vatistas);
        CHECK_CUDA(cudaGetLastError());

        const Dim3<u64> n2{lhs.layout.m(), colloc.layout.ns(i)};
        // last wing row
        kernel_influence<block><<<grid_size(block, n2)(), block()>>>(n2.x, n2.y, lhs_section + end_wing * lhs.layout.stride(), colloc.ptr, colloc.layout.stride(), vwing_section + (verts_wing.layout.nc(i)-1)*verts_wing.layout.ns(i), verts_wing.layout.stride(), verts_wing.layout.ns(i), normals.ptr, normals.layout.stride(), sigma_vatistas);
        CHECK_CUDA(cudaGetLastError());

        while (condition[i] < iteration) {
            const Dim3<u64> n3{lhs.layout.m(), colloc.layout.ns(i)};
            // each wake row
            kernel_influence<block><<<grid_size(block, n3)(), block()>>>(
                n3.x,
                n3.y,
                lhs_section + end_wing * lhs.layout.stride(),
                colloc.ptr, colloc.layout.stride(),
                vwake_section + (verts_wake.layout.nc(i) - condition[i] - 2) * verts_wake.layout.ns(i),
                verts_wake.layout.stride(), verts_wake.layout.ns(i), 
                normals.ptr, normals.layout.stride(),
                sigma_vatistas
            );
            condition[i]++;
        }
    }
}

void BackendCUDA::compute_rhs(const FlowData& flow) {
    default_backend.compute_rhs(flow);
}

void BackendCUDA::compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) {
    default_backend.compute_rhs(flow, section_alphas);
}

void BackendCUDA::lu_factor() {
    tiny::ScopedTimer timer("Factor");
    int n = (int)mesh.nb_panels_wing();
    int h_info = 0;

    CHECK_CUSOLVER(cusolverDnSgetrf(CtxManager::getInstance().cusolver(), n, n, d_lhs, n, d_solver_buffer, d_solver_ipiv, d_solver_info));
    CHECK_CUDA(cudaMemcpy(&h_info, d_solver_info, sizeof(int), cudaMemcpyDeviceToHost)); // sync
    if (h_info != 0) printf("Error: LU factorization failed\n");
};

void BackendCUDA::lu_solve() {
    tiny::ScopedTimer timer("Solve");
    //default_backend.solve();
    int n = (int)mesh.nb_panels_wing();
    int h_info = 0;

    // copy data to device (temporary)
    CHECK_CUDA(cudaMemcpy(d_rhs, default_backend.rhs.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Solve on device
    CHECK_CUSOLVER(cusolverDnSgetrs(CtxManager::getInstance().cusolver(), CUBLAS_OP_N, n, 1, d_lhs, n, d_solver_ipiv, d_rhs, n, d_solver_info));
    CHECK_CUDA(cudaMemcpy(&h_info, d_solver_info, sizeof(int), cudaMemcpyDeviceToHost)); // sync
    if (h_info != 0) printf("Error: LU solve failed\n");

    // copy data back to host
    CHECK_CUDA(cudaMemcpy(default_backend.gamma.data(), d_rhs, n * sizeof(float), cudaMemcpyDeviceToHost));
}

f32 BackendCUDA::compute_coefficient_cl(
    const FlowData& flow,
    const f32 area,
    const u64 j,
    const u64 n) {
    return default_backend.compute_coefficient_cl(flow, area, j, n);
}

f32 BackendCUDA::compute_coefficient_cd(
    const FlowData& flow,
    const f32 area,
    const u64 j,
    const u64 n) {
    return default_backend.compute_coefficient_cd(flow, area, j, n);
}

linalg::alias::float3 BackendCUDA::compute_coefficient_cm(
    const FlowData& flow,
    const f32 area,
    const f32 chord,
    const u64 j,
    const u64 n) {
    return default_backend.compute_coefficient_cm(flow, area, chord, j, n);
}

void BackendCUDA::compute_delta_gamma() {
    default_backend.compute_delta_gamma();
}

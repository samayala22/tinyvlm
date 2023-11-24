#include "vlm_backend_avx2.hpp"
#include "vlm_backend_cuda.hpp"

#include "simpletimer.hpp"

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

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
    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    std::printf("----- CUDA Device information -----\n");
    std::printf("Found %d device(s)\n", deviceCount);

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

protected:
    cusolverDnHandle_t m_cusolver = nullptr;
    cudaStream_t m_stream = nullptr;
    cublasHandle_t m_cublas = nullptr;

private:
    CtxManager() = default;
    ~CtxManager() = default;
};

BackendCUDA::BackendCUDA(Mesh& mesh, Data& data) : default_backend(mesh, data), Backend(mesh, data) {
    printCudaInfo();
    auto& ctx = CtxManager::getInstance();
    ctx.create();

    u64 n = (u64)mesh.nb_panels_wing();
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_lhs, n*n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_rhs, n * sizeof(float)));
}

BackendCUDA::~BackendCUDA() {
    auto& ctx = CtxManager::getInstance();
    ctx.destroy();

    CHECK_CUDA(cudaFree(d_lhs));
    CHECK_CUDA(cudaFree(d_rhs));
}

// For the moment, cuda backend just falls back to AVX2

void BackendCUDA::reset() {
    default_backend.reset();
}

void BackendCUDA::compute_lhs() {
    default_backend.compute_lhs();
}

void BackendCUDA::compute_rhs() {
    default_backend.compute_rhs();
}

int CUDA_LU_solver(cusolverDnHandle_t handle, float *d_A,
                   float *d_b, int n) {
    // All pointers are device pointers
    // A is column major 
    // Ax = b -> result is stored in b
    int bufferSize = 0;
    int *info = NULL;
    float *buffer = NULL;
    int *ipiv = NULL;  // pivoting sequence
    int h_info = 0;

    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(handle, n, n, (float *)d_A,
                                                n, &bufferSize));

    CHECK_CUDA(cudaMalloc((void**)&info, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&buffer, sizeof(float) * bufferSize));
    CHECK_CUDA(cudaMalloc((void**)&ipiv, sizeof(int) * n));

    CHECK_CUDA(cudaMemset(info, 0, sizeof(int)));

    CHECK_CUSOLVER(cusolverDnSgetrf(handle, n, n, d_A, n, buffer, ipiv, info));
    CHECK_CUDA(
        cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
    fprintf(stderr, "Error: LU factorization failed\n");
    }

    CHECK_CUSOLVER(
        cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_A, n, ipiv, d_b, n, info));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (info) {
    CHECK_CUDA(cudaFree(info));
    }
    if (buffer) {
    CHECK_CUDA(cudaFree(buffer));
    }
    if (ipiv) {
    CHECK_CUDA(cudaFree(ipiv));
    }

    return 0;
}

void BackendCUDA::solve() {
    SimpleTimer timer("Solve");
    //default_backend.solve();
    auto& ctx = CtxManager::getInstance();
    u64 N = (u64)mesh.nb_panels_wing();

    // copy data to device
    CHECK_CUDA(cudaMemcpy(d_lhs, default_backend.lhs.data(), N*N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, default_backend.rhs.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Solve on device
    CUDA_LU_solver(ctx.cusolver(), d_lhs, d_rhs, N);

    // copy data back to host
    CHECK_CUDA(cudaMemcpy(data.gamma.data(), d_rhs, N * sizeof(float), cudaMemcpyDeviceToHost));
}

void BackendCUDA::compute_forces() {
    default_backend.compute_forces();
}

void BackendCUDA::compute_delta_gamma() {
    default_backend.compute_delta_gamma();
}

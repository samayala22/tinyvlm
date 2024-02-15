#include "vlm_backend_cpu.hpp"
#include "vlm_backend_cuda.hpp"

#include "simpletimer.hpp"

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_math.h"

#include <cstdio>
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

protected:
    cusolverDnHandle_t m_cusolver = nullptr;
    cudaStream_t m_stream = nullptr;
    cublasHandle_t m_cublas = nullptr;

private:
    CtxManager() = default;
    ~CtxManager() = default;
};

BackendCUDA::BackendCUDA(Mesh& mesh) : default_backend(mesh), Backend(mesh) {
    printCudaInfo();
    auto& ctx = CtxManager::getInstance();
    ctx.create();

    u64 n = mesh.nb_panels_wing();
    u64 npt = mesh.nb_panels_total();
    u64 nvt = mesh.nb_vertices_total();
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_lhs, n*n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_rhs, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_gamma, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_delta_gamma, n * sizeof(float)));

    h_mesh.nb_panels = n;
    h_mesh.ns = mesh.ns;
    h_mesh.nc = mesh.nc;
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.v.x, nvt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.v.y, nvt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.v.z, nvt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.colloc.x, npt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.colloc.y, npt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.colloc.z, npt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.normal.x, npt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.normal.y, npt * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&h_mesh.normal.z, npt * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_mesh.v.x, mesh.v.x.data(), nvt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.v.y, mesh.v.y.data(), nvt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.v.z, mesh.v.z.data(), nvt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.colloc.x, mesh.colloc.x.data(), npt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.colloc.y, mesh.colloc.y.data(), npt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.colloc.z, mesh.colloc.z.data(), npt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.normal.x, mesh.normal.x.data(), npt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.normal.y, mesh.normal.y.data(), npt * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_mesh.normal.z, mesh.normal.z.data(), npt * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMalloc((void**)&d_mesh, sizeof(MeshProxy)));      
    CHECK_CUDA(cudaMemcpy(d_mesh, &h_mesh, sizeof(MeshProxy), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
}

BackendCUDA::~BackendCUDA() {
    auto& ctx = CtxManager::getInstance();
    ctx.destroy();

    CHECK_CUDA(cudaFree(d_lhs));
    CHECK_CUDA(cudaFree(d_rhs));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_delta_gamma));
    CHECK_CUDA(cudaFree(h_mesh.v.x));
    CHECK_CUDA(cudaFree(h_mesh.v.y));
    CHECK_CUDA(cudaFree(h_mesh.v.z));
    CHECK_CUDA(cudaFree(h_mesh.colloc.x));
    CHECK_CUDA(cudaFree(h_mesh.colloc.y)); 
    CHECK_CUDA(cudaFree(h_mesh.colloc.z));
    CHECK_CUDA(cudaFree(h_mesh.normal.x));
    CHECK_CUDA(cudaFree(h_mesh.normal.y));
    CHECK_CUDA(cudaFree(h_mesh.normal.z));
    CHECK_CUDA(cudaFree(d_mesh));
}

// For the moment, cuda backend just falls back to cpu backend

void BackendCUDA::reset() {
    default_backend.reset();
    u64 n = mesh.nb_panels_wing();

    CHECK_CUDA(cudaMemset(d_lhs, 0, n * n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_rhs, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_gamma, 0, n * sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());
}

#define RCUT 1e-10f
#define RCUT2 1e-5f

#define PI_f 3.141593f

__device__ inline float3 kernel_biosavart(float3& colloc, const float3& vertex1, const float3& vertex2, const float& sigma) {
    float3 r0 = vertex2 - vertex1;
    float3 r1 = colloc - vertex1;
    float3 r2 = colloc - vertex2;
    // Katz Plotkin, Low speed Aero | Eq 10.115
    float3 r1r2cross = cross(r1, r2);
    float r1_norm = length(r1);
    float r2_norm = length(r2);
    float square = length2(r1r2cross);
    
    if ((square<RCUT) || (r1_norm<RCUT2) || (r2_norm<RCUT2)) {
        float3 res = {0.0f, 0.0f, 0.0f};
        return res;
    }

    float smoother = sigma*sigma*length2(r0);

    float coeff = (dot(r0,r1)*r2_norm - dot(r0, r2)*r1_norm) / (4.0f*PI_f*sqrt(square*square + smoother*smoother)*r1_norm*r2_norm);
    return r1r2cross * coeff;
}

__device__ inline void kernel_symmetry(float3& inf, float3 colloc, const float3& vertex0, const float3& vertex1, const float& sigma) {
    float3 induced_speed = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf.x += induced_speed.x;
    inf.y += induced_speed.y;
    inf.z += induced_speed.z;
    colloc.y = -colloc.y; // wing symmetry
    float3 induced_speed_sym = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf.x += induced_speed_sym.x;
    inf.y -= induced_speed_sym.y;
    inf.z += induced_speed_sym.z;
}

__global__ void kernel_influence_cuda(const MeshProxy* m, float* d_lhs, const uint64_t lidx, const float sigma) {
    u64 ia = blockIdx.y * blockDim.y + threadIdx.y;
    u64 ia2 = blockIdx.x * blockDim.x + threadIdx.x;

    if (ia2 >= m->nb_panels * m->nb_panels) return;

    // if (ia > lidx || ia2 > m.nb_panels) {
    //     return;
    // }

    // const u64 v0 = ia + ia / m.ns;
    // const u64 v1 = v0 + 1;
    // const u64 v3 = v0 + m.ns + 1;
    // const u64 v2 = v3 + 1;
    // const float3 vertex0{m.v.x[v0], m.v.y[v0], m.v.z[v0]};
    // const float3 vertex1{m.v.x[v1], m.v.y[v1], m.v.z[v1]};
    // const float3 vertex2{m.v.x[v2], m.v.y[v2], m.v.z[v2]};
    // const float3 vertex3{m.v.x[v3], m.v.y[v3], m.v.z[v3]};

    // const float3 colloc = {m.colloc.x[ia2], m.colloc.y[ia2], m.colloc.z[ia2]};
    // const float3 normal = {m.normal.x[ia2], m.normal.y[ia2], m.normal.z[ia2]};
    // float3 inf{0.0f, 0.0f, 0.0f};
    // kernel_symmetry(inf, colloc, vertex0, vertex1, sigma);
    // kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
    // kernel_symmetry(inf, colloc, vertex2, vertex3, sigma);
    // kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
    // d_lhs[ia * m.nb_panels + ia2] += dot(inf, normal);
    d_lhs[ia2] = static_cast<float>(ia2);
}

void BackendCUDA::compute_lhs(const FlowData& flow) {
    default_backend.compute_lhs(flow);
    const u64 block_size = 1024;
    const u64 lidx = (mesh.nc - 1) * mesh.ns;
    const u64 grid_size = (lidx + block_size - 1) / block_size;
    kernel_influence_cuda<<<grid_size, block_size>>>(d_mesh, d_lhs, lidx, flow.sigma_vatistas);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(default_backend.lhs_dummy.data(), d_lhs, mesh.nb_panels_wing() * mesh.nb_panels_wing() * sizeof(float), cudaMemcpyDeviceToHost));
    for (u64 i = 0; i < mesh.nb_panels_wing(); i++) {
        std::printf("CPU: %e %e GPU\n", default_backend.lhs[i], default_backend.lhs_dummy[i]);
    }
}

void BackendCUDA::compute_rhs(const FlowData& flow) {
    default_backend.compute_rhs(flow);
}

void BackendCUDA::compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) {
    default_backend.compute_rhs(flow, section_alphas);
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

void BackendCUDA::lu_factor() {};

void BackendCUDA::lu_solve() {
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
    CHECK_CUDA(cudaMemcpy(default_backend.gamma.data(), d_rhs, N * sizeof(float), cudaMemcpyDeviceToHost));
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

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

/// @brief Memory manager implementation for the CUDA backend
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
                    kernel_fill_f32<block.x><<<grid_size(block, n)(), block()>>>(ptr, value, size);
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
    print_cuda_info();
    CtxManager::getInstance().create();
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
        kernel_gamma_delta<block.x, block.y><<<grid_size(block, n)(), block()>>>(n.x, n.y, s_gamma_delta + surf.ns, s_gamma + surf.ns);
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
void BackendCUDA::lhs_assemble(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& verts_wake, std::vector<u32>& condition, u32 iteration) {
    assert(condition.size() == colloc.layout.surfaces().size());
    std::fill(condition.begin(), condition.end(), 0); // reset conditon increment vars

    for (u32 i = 0; i < colloc.layout.surfaces().size(); i++) {
        f32* lhs_section = lhs.ptr + colloc.layout.offset(i) * lhs.layout.stride();
        
        f32* vwing_section = verts_wing.ptr + verts_wing.layout.offset(i);
        f32* vwake_section = verts_wake.ptr + verts_wake.layout.offset(i);
        const u64 end_wing = (colloc.layout.nc(i) - 1) * colloc.layout.ns(i);
        
        constexpr Dim3<u32> block{32, 16};
        const Dim3<u64> n{lhs.layout.m(), end_wing};
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

        const Dim3<u64> n2{lhs.layout.m(), colloc.layout.ns(i)};
        // last wing row
        kernel_influence<block.x, block.y><<<grid_size(block, n2)(), block()>>>(
            n2.x,
            n2.y, 
            lhs_section + end_wing * lhs.layout.stride(),
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
            const Dim3<u64> n3{lhs.layout.m(), colloc.layout.ns(i)};
            // each wake row
            kernel_influence<block.x, block.y><<<grid_size(block, n3)(), block()>>>(
                n3.x,
                n3.y,
                lhs_section + end_wing * lhs.layout.stride(),
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
void BackendCUDA::rhs_assemble_velocities(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& velocities) {
    // const tiny::ScopedTimer timer("RHS");
    assert(rhs.layout.stride() == rhs.size()); // single dim
    assert(rhs.layout.stride() == normals.layout.stride());
    assert(rhs.layout.stride() == velocities.layout.stride());
    assert(rhs.layout.dims() == 1);

    constexpr Dim3<u32> block{768};
    const Dim3<u64> n{rhs.size()};
    kernel_rhs_assemble_velocities<block.x><<<grid_size(block, n)(), block()>>>(n.x, rhs.ptr, velocities.ptr, velocities.layout.stride(), normals.ptr, normals.layout.stride());
}

void BackendCUDA::rhs_assemble_wake_influence(View<f32, MultiSurface>& rhs, const View<f32, MultiSurface>& gamma_wake, const View<f32, MultiSurface>& colloc, const View<f32, MultiSurface>& normals, const View<f32, MultiSurface>& verts_wake, u32 iteration) {
    // TODO
}

void BackendCUDA::displace_wake_rollup(View<f32, MultiSurface>& wake_rollup, const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wake, f32 dt, u32 iteration) {
    // TODO
}

void BackendCUDA::gamma_shed(View<f32, MultiSurface>& gamma_wing, View<f32, MultiSurface>& gamma_wing_prev, View<f32, MultiSurface>& gamma_wake, u32 iteration) {
    // const tiny::ScopedTimer timer("Shed Gamma");

    memory->copy(MemoryTransfer::DeviceToDevice, gamma_wing_prev.ptr, gamma_wing.ptr, gamma_wing.size_bytes());
    for (u64 i = 0; i < gamma_wake.layout.surfaces().size(); i++) {
        assert(iteration < gamma_wake.layout.nc(i));
        f32* gamma_wake_ptr = gamma_wake.ptr + gamma_wake.layout.offset(i) + (gamma_wake.layout.nc(i) - iteration - 1) * gamma_wake.layout.ns(i);
        f32* gamma_wing_ptr = gamma_wing.ptr + gamma_wing.layout.offset(i) + (gamma_wing.layout.nc(i) - 1) * gamma_wing.layout.ns(i); // last row
        memory->copy(MemoryTransfer::DeviceToDevice, gamma_wake_ptr, gamma_wing_ptr, gamma_wing.layout.ns(i) * sizeof(f32));
    }
}

void BackendCUDA::lu_allocate(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) {
    int bufsize = 0;
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(CtxManager::getInstance().cusolver(), lhs.layout.m(), lhs.layout.n(), lhs.ptr, lhs.layout.stride(), &bufsize));
    d_solver_info = (i32*)memory->alloc(MemoryLocation::Device, sizeof(i32));
    d_solver_buffer = (f32*)memory->alloc(MemoryLocation::Device, sizeof(f32) * bufsize);
    d_solver_ipiv = (i32*)memory->alloc(MemoryLocation::Device, sizeof(i32) * lhs.layout.n());
}

void BackendCUDA::lu_factor(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs) {
    // const tiny::ScopedTimer timer("Factor");
    assert(lhs.layout.m() == lhs.layout.n()); // square matrix
    const int32_t n = static_cast<int32_t>(lhs.layout.n());
    int h_info = 0;

    CHECK_CUSOLVER(cusolverDnSgetrf(CtxManager::getInstance().cusolver(), n, n, lhs.ptr, n, d_solver_buffer, d_solver_ipiv, d_solver_info));
    memory->copy(MemoryTransfer::DeviceToHost, &h_info, d_solver_info, sizeof(int));
    if (h_info != 0) printf("Error: LU factorization failed\n");
};

void BackendCUDA::lu_solve(View<f32, Matrix<MatrixLayout::ColMajor>>& lhs, View<f32, MultiSurface>& rhs, View<f32, MultiSurface>& gamma) {
    // tiny::ScopedTimer timer("Solve");
    //default_backend.solve();
    const i32 n = static_cast<int32_t>(lhs.layout.n());
    i32 h_info = 0;

    memory->copy(MemoryTransfer::DeviceToDevice, gamma.ptr, rhs.ptr, rhs.size_bytes());
    CHECK_CUSOLVER(cusolverDnSgetrs(CtxManager::getInstance().cusolver(), CUBLAS_OP_N, n, 1, lhs.ptr, n, d_solver_ipiv, gamma.ptr, n, d_solver_info));
    memory->copy(MemoryTransfer::DeviceToHost, &h_info, d_solver_info, sizeof(int));
    if (h_info != 0) printf("Error: LU solve failed\n");
}

f32 BackendCUDA::coeff_steady_cl_single(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& gamma_delta, const FlowData& flow, f32 area) {
    f32 h_cl = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<u32> block{32, 16};
    const Dim3<u64> n{gamma_delta.layout.ns(), gamma_delta.layout.nc()};
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
    memory->copy(MemoryTransfer::DeviceToHost, &h_cl, d_val, sizeof(f32));
    CtxManager::getInstance().sync();
    
    return h_cl / (0.5f * flow.rho * linalg::length2(flow.freestream) * area);
}

f32 BackendCUDA::coeff_steady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_delta, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    f32 cl = 0.0f;
    f32 total_area = 0.0f;
    for (u64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
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
    // TODO
    return 0.0f;
}

f32 BackendCUDA::coeff_unsteady_cl_multi(const View<f32, MultiSurface>& verts_wing, const View<f32, MultiSurface>& gamma_wing_delta, const View<f32, MultiSurface>& gamma_wing, const View<f32, MultiSurface>& gamma_wing_prev, const View<f32, MultiSurface>& velocities, const View<f32, MultiSurface>& areas, const View<f32, MultiSurface>& normals, const linalg::alias::float3& freestream, f32 dt) {
    // TODO
    return 0.0f;
}

f32 BackendCUDA::coeff_steady_cd_single(const View<f32, SingleSurface>& verts_wake, const View<f32, SingleSurface>& gamma_wake, const FlowData& flow, f32 area) {
    f32 h_cd = 0.0f;
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<u32> block{64, 8};
    const Dim3<u64> n{verts_wake.layout.ns()-1, verts_wake.layout.ns()-1};
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
    memory->copy(MemoryTransfer::DeviceToHost, &h_cd, d_val, sizeof(f32));
    CtxManager::getInstance().sync();
    return h_cd / (linalg::length2(flow.freestream) * area);
}

// TODO: move in backend.cpp ?
f32 BackendCUDA::coeff_steady_cd_multi(const View<f32, MultiSurface>& verts_wake, const View<f32, MultiSurface>& gamma_wake, const FlowData& flow, const View<f32, MultiSurface>& areas) {
    // const tiny::ScopedTimer timer("Compute CL");
    f32 cd = 0.0f;
    f32 total_area = 0.0f;
    for (u64 i = 0; i < verts_wake.layout.surfaces().size(); i++) {
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

void BackendCUDA::mesh_metrics(const f32 alpha_rad, const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& colloc, View<f32, MultiSurface>& normals, View<f32, MultiSurface>& areas) {
    for (int m = 0; m < colloc.layout.surfaces().size(); m++) {
        const f32* verts_wing_ptr = verts_wing.ptr + verts_wing.layout.offset(m);
        f32* colloc_ptr = colloc.ptr + colloc.layout.offset(m);
        f32* normals_ptr = normals.ptr + normals.layout.offset(m);
        f32* areas_ptr = areas.ptr + areas.layout.offset(m);

        constexpr Dim3<u32> block{24, 32}; // ns, nc
        const Dim3<u64> n{colloc.layout.ns(m), colloc.layout.nc(m)};
        kernel_mesh_metrics<block.x, block.y><<<grid_size(block, n)(), block()>>>(n.y, n.x, colloc_ptr, colloc.layout.stride(), normals_ptr, normals.layout.stride(), areas_ptr, verts_wing_ptr, verts_wing.layout.stride(), alpha_rad);
    }
}

f32 BackendCUDA::mesh_mac(const View<f32, SingleSurface>& verts_wing, const View<f32, SingleSurface>& areas) {
    // TODO
    return 0.0f;
}

void BackendCUDA::displace_wing(const View<f32, Tensor<3>>& transforms, View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wing_init) {
    // const tiny::ScopedTimer t("Mesh::move");
    assert(transforms.layout.shape(2) == verts_wing.layout.surfaces().size());
    assert(verts_wing.layout.size() == verts_wing_init.layout.size());

    const f32 alpha = 1.0f;
    const f32 beta = 0.0f;

    // TODO: parallel for
    for (u64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
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
            transforms.ptr + transforms.layout.stride(2)*i,
            4,
            &beta,
            verts_wing.ptr + verts_wing.layout.offset(i),
            static_cast<int>(verts_wing.layout.stride())
        ));
    }
}

void BackendCUDA::wake_shed(const View<f32, MultiSurface>& verts_wing, View<f32, MultiSurface>& verts_wake, u32 iteration) {
    assert(verts_wing.layout.surfaces().size() == verts_wake.layout.surfaces().size());
    for (u64 i = 0; i < verts_wing.layout.surfaces().size(); i++) {
        assert(iteration < verts_wake.layout.nc(i));
        f32* vwing = verts_wing.ptr + verts_wing.layout.offset(i) + (verts_wing.layout.nc(i) - 1) * verts_wing.layout.ns(i);
        f32* vwake = verts_wake.ptr + verts_wake.layout.offset(i) + (verts_wake.layout.nc(i) - iteration - 1) * verts_wake.layout.ns(i);

        memory->copy(MemoryTransfer::DeviceToDevice, vwake + 0*verts_wake.layout.stride(), vwing + 0*verts_wing.layout.stride(), verts_wing.layout.ns(i) * sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, vwake + 1*verts_wake.layout.stride(), vwing + 1*verts_wing.layout.stride(), verts_wing.layout.ns(i) * sizeof(f32));
        memory->copy(MemoryTransfer::DeviceToDevice, vwake + 2*verts_wake.layout.stride(), vwing + 2*verts_wing.layout.stride(), verts_wing.layout.ns(i) * sizeof(f32));
    }
}

f32 BackendCUDA::mesh_area(const View<f32, SingleSurface>& areas) {
    cudaMemset(d_val, 0, sizeof(f32));
    constexpr Dim3<u32> block{512}; // ns, nc
    const Dim3<u64> n{areas.layout.size()};
    kernel_reduce<<<grid_size(block, n)(), block()>>>(n.x, areas.ptr, d_val);
    CHECK_CUDA(cudaGetLastError());
    f32 h_area;
    memory->copy(MemoryTransfer::DeviceToHost, &h_area, d_val, sizeof(f32));
    return h_area;
}
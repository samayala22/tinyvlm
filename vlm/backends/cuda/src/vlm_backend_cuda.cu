// VLM
#include "vlm_backend_cuda.hpp"
#include "vlm_backend_cuda_kernels.cuh"
#include "vlm_executor.hpp"

// External
#include "tinytimer.hpp"
#include <taskflow/cuda/cudaflow.hpp>

// CUDA
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
        // Create CUDA stream
        CHECK_CUDA(cudaStreamCreate(&m_stream));
    }

    void destroy() {
        CHECK_CUDA(cudaStreamDestroy(m_stream)); // destroy stream last
    }

    cudaStream_t stream() {return(m_stream);}
    void sync() {CHECK_CUDA(cudaStreamSynchronize(m_stream));}

private:
    cudaStream_t m_stream = nullptr;
    
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
    name = "CUDA";
    print_cuda_info();
    CtxManager::getInstance().create();
    d_cl = (f32*)memory->alloc(Location::Device, sizeof(f32));
    d_cd = (f32*)memory->alloc(Location::Device, sizeof(f32));
    d_cm_x = (f32*)memory->alloc(Location::Device, sizeof(f32));
    d_cm_y = (f32*)memory->alloc(Location::Device, sizeof(f32));
    d_cm_z = (f32*)memory->alloc(Location::Device, sizeof(f32));
    d_mac = (f32*)memory->alloc(Location::Device, sizeof(f32));
}

BackendCUDA::~BackendCUDA() {
    memory->free(Location::Device, d_cl);
    memory->free(Location::Device, d_cd);
    memory->free(Location::Device, d_cm_x);
    memory->free(Location::Device, d_cm_y);
    memory->free(Location::Device, d_cm_z);
    memory->free(Location::Device, d_mac);
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
void BackendCUDA::lhs_assemble(TensorView2fD& lhs, MultiTensorView3fD& colloc, MultiTensorView3fD& normals, MultiTensorView3fD& verts_wing, MultiTensorView3fD& verts_wake, std::vector<i32>& condition, i32 iteration) {
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
            i64 condition_idx = m_i + m_j * static_cast<i64>(colloc.size());
            auto colloc_i = colloc[m_i];
            auto colloc_j = colloc[m_j];
            auto normals_i = normals[m_i];
            auto verts_wing_j = verts_wing[m_j];
            auto verts_wake_j = verts_wake[m_j];

            f32* lhs_section = lhs.ptr() + offset_i + offset_j * lhs.stride(1);

            i64 zero = 0;
            i64 end_wing = (colloc_j.shape(1) - 1) * colloc_j.shape(0);
            
            auto wing_pass = graph.emplace([=](){
                Dim3<i64> n{colloc_i.stride(2), end_wing};
                kernel_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
                    n.x,
                    n.y,
                    lhs_section,
                    lhs.stride(1),
                    colloc_i.ptr(),
                    colloc_i.stride(2),
                    verts_wing_j.ptr(),
                    verts_wing_j.stride(2),
                    verts_wing_j.stride(1),
                    normals_i.ptr(),
                    normals_i.stride(2),
                    sigma_vatistas
                );
                CHECK_CUDA(cudaGetLastError());
            }).name("wing pass");

            auto last_row = graph.emplace([=](){
                Dim3<i64> n{colloc_i.stride(2), colloc_j.shape(0)};
                kernel_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
                    n.x,
                    n.y,
                    lhs_section + end_wing * lhs.stride(1),
                    lhs.stride(1),
                    colloc_i.ptr(),
                    colloc_i.stride(2),
                    verts_wing_j.ptr() + verts_wing_j.offset({0, -2, 0}),
                    verts_wing_j.stride(2),
                    verts_wing_j.stride(1),
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
                Dim3<i64> n{colloc_i.stride(2), colloc_j.shape(0)};
                kernel_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
                    n.x,
                    n.y,
                    lhs_section + end_wing * lhs.stride(1),
                    lhs.stride(1),
                    colloc_i.ptr(),
                    colloc_i.stride(2),
                    verts_wake_j.ptr() + verts_wake_j.offset({0, -2-condition[condition_idx], 0}),
                    verts_wake_j.stride(2),
                    verts_wake_j.stride(1),
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
void BackendCUDA::rhs_assemble_velocities(TensorView1fD& rhs, MultiTensorView3fD& normals, MultiTensorView3fD& velocities) {
    // tiny::ScopedTimer timer("RHS");
    
    i64 offset = 0;
    for (i64 m = 0; m < normals.size(); m++) {
        assert(offset <= rhs.size());
        auto& normals_i = normals[m];
        auto& velocities_m = velocities[m];
        constexpr Dim3<i32> block{768};
        Dim3<i64> n{normals_i.stride(2)};
        kernel_rhs_assemble_velocities<block.x><<<grid_size(block, n)(), block()>>>(
            n.x,
            rhs.ptr() + offset,
            velocities_m.ptr(),
            velocities_m.stride(2),
            normals_i.ptr(),
            normals_i.stride(2)
        );
        offset += normals_i.stride(2); // Note: using stride here is not technically correct
    }
}

void BackendCUDA::rhs_assemble_wake_influence(TensorView1fD& rhs, MultiTensorView2fD& gamma_wake, MultiTensorView3fD& colloc, MultiTensorView3fD& normals, MultiTensorView3fD& verts_wake, std::vector<bool>& lifting, i32 iteration) {
    if (iteration == 0) return; // cuda doesnt support 0 sized domain
    constexpr Dim3<i32> block{32, 16}; // TODO: do not modify this 

    i64 offset = 0;
    for (i32 m_i = 0; m_i < normals.size(); m_i++) {
        i64 m_i_num_panels = normals[m_i].shape(0) * normals[m_i].shape(1);
        for (i32 m_j = 0; m_j < normals.size(); m_j++) {
            if (!lifting[m_j]) continue;
            auto& colloc_i = colloc[m_i];
            auto& normals_i = normals[m_i];
            auto& gamma_wake_j = gamma_wake[m_j];
            auto& verts_wake_j = verts_wake[m_j];
            i64 wake_m  = iteration; 
            i64 wake_n  = verts_wake_j.shape(0) - 1; // spanwise number of wake panels
            Dim3<i64> n{wake_m * wake_n, m_i_num_panels};
            kernel_wake_influence<block.x, block.y><<<grid_size(block, n)(), block()>>>(
                wake_m, 
                wake_n,
                m_i_num_panels,
                colloc_i.ptr(),
                colloc_i.stride(2),
                normals_i.ptr(),
                normals_i.stride(2),
                verts_wake_j.ptr() + verts_wake_j.offset({0, -1-iteration, 0}),
                verts_wake_j.stride(2),
                gamma_wake_j.ptr() + gamma_wake_j.offset({0, gamma_wake_j.shape(1) - iteration}),
                rhs.ptr() + offset,
                sigma_vatistas
            );
            CHECK_CUDA(cudaGetLastError());
        }
        offset += m_i_num_panels;
    }
}

void BackendCUDA::forces_unsteady(
    TensorView3fD& verts_wing,
    TensorView2fD& gamma_delta,
    TensorView2fD& gamma_dt,
    TensorView3fD& velocities,
    TensorView2fD& areas,
    TensorView3fD& normals,
    TensorView3fD& forces
)
{
    constexpr Dim3<i32> block{32, 16};
    Dim3<i64> n{gamma_delta.shape(0), gamma_delta.shape(1)};
    DataDims dims;
    dims.panel_shape_0 = normals.shape(0);
    dims.panel_shape_1 = normals.shape(1);
    dims.panel_stride_1 = normals.stride(1);
    dims.panel_stride_2 = normals.stride(2);
    dims.vertex_shape_0 = verts_wing.shape(0);
    dims.vertex_shape_1 = verts_wing.shape(1);
    dims.vertex_stride_1 = verts_wing.stride(1);
    dims.vertex_stride_2 = verts_wing.stride(2);
    kernel_forces_unsteady<<<grid_size(block, n)(), block()>>>(
        dims,
        verts_wing.ptr(),
        gamma_delta.ptr(),
        gamma_dt.ptr(),
        velocities.ptr(),
        areas.ptr(),
        normals.ptr(),
        forces.ptr()
    );
    CHECK_CUDA(cudaGetLastError());
    // stream sync
}

f32 BackendCUDA::coeff_cl(
    TensorView3fD& forces,
    linalg::float3& lift_axis,
    linalg::float3& freestream,
    f32 rho,
    f32 area
)
{
    cudaMemset(d_cl, 0, sizeof(f32));
    constexpr Dim3<i32> block{32, 16};
    Dim3<i64> n{forces.shape(0), forces.shape(1)};
    DataDims dims;
    dims.panel_shape_0 = forces.shape(0);
    dims.panel_shape_1 = forces.shape(1);
    dims.panel_stride_1 = forces.stride(1);
    dims.panel_stride_2 = forces.stride(2);
    kernel_coeff_cl<<<grid_size(block, n)(), block()>>>(
        dims,
        forces.ptr(),
        float3{lift_axis.x, lift_axis.y, lift_axis.z},
        d_cl
    );
    CHECK_CUDA(cudaGetLastError());
    // stream sync
    f32 h_cl = 0.0f;
    memory->copy(Location::Host, &h_cl, 1, Location::Device, d_cl, 1, sizeof(f32), 1);
    return h_cl / (0.5f * rho * linalg::length2(freestream) * area);
}

linalg::float3 BackendCUDA::coeff_cm(
    TensorView3fD& forces,
    TensorView3fD& verts_wing,
    linalg::float3& ref_pt,
    linalg::float3& freestream,
    f32 rho,
    f32 area,
    f32 mac
)
{
    cudaMemset(d_cm_x, 0, sizeof(f32));
    cudaMemset(d_cm_y, 0, sizeof(f32));
    cudaMemset(d_cm_z, 0, sizeof(f32));

    constexpr Dim3<i32> block{32, 16};
    Dim3<i64> n{forces.shape(0), forces.shape(1)};
    DataDims dims;
    dims.panel_shape_0 = forces.shape(0);
    dims.panel_shape_1 = forces.shape(1);
    dims.panel_stride_1 = forces.stride(1);
    dims.panel_stride_2 = forces.stride(2);
    dims.vertex_shape_0 = verts_wing.shape(0);
    dims.vertex_shape_1 = verts_wing.shape(1);
    dims.vertex_stride_1 = verts_wing.stride(1);
    dims.vertex_stride_2 = verts_wing.stride(2);
    kernel_coeff_cm<<<grid_size(block, n)(), block()>>>(
        dims,
        forces.ptr(),
        verts_wing.ptr(),
        float3{ref_pt.x, ref_pt.y, ref_pt.z},
        d_cm_x,
        d_cm_y,
        d_cm_z
    );
    CHECK_CUDA(cudaGetLastError());
    // stream sync
    linalg::float3 h_cm;
    memory->copy(Location::Host, &h_cm.x, 1, Location::Device, d_cm_x, 1, sizeof(f32), 1);
    memory->copy(Location::Host, &h_cm.y, 1, Location::Device, d_cm_y, 1, sizeof(f32), 1);
    memory->copy(Location::Host, &h_cm.z, 1, Location::Device, d_cm_z, 1, sizeof(f32), 1);
    
    return h_cm / (0.5f * rho * linalg::length2(freestream) * area * mac);
}

void BackendCUDA::mesh_metrics(f32 alpha_rad, MultiTensorView3fD& verts_wing, MultiTensorView3fD& colloc, MultiTensorView3fD& normals, MultiTensorView2fD& areas) {
    for (int m = 0; m < colloc.size(); m++) {
        auto& colloc_i = colloc[m];
        auto& normals_i = normals[m];
        auto& areas_i = areas[m];
        auto& verts_wing_m = verts_wing[m];

        constexpr Dim3<i32> block{32, 16}; // ns, nc
        Dim3<i64> n{colloc_i.shape(0), colloc_i.shape(1)};
        kernel_mesh_metrics<block.x, block.y><<<grid_size(block, n)(), block()>>>(
            n.y,
            n.x,
            colloc_i.ptr(),
            colloc_i.stride(2),
            normals_i.ptr(),
            normals_i.stride(2),
            areas_i.ptr(),
            verts_wing_m.ptr(),
            verts_wing_m.stride(2),
            alpha_rad
        );
    }
}

f32 BackendCUDA::mesh_mac(
    TensorView3fD& verts_wing,
    TensorView2fD& areas
) 
{
    cudaMemset(d_mac, 0, sizeof(f32));
    constexpr Dim3<i32> block{768};
    Dim3<i64> n{areas.shape(0)};
    DataDims dims;
    dims.panel_shape_0 = areas.shape(0);
    dims.panel_shape_1 = areas.shape(1);
    dims.vertex_shape_0 = verts_wing.shape(0);
    dims.vertex_shape_1 = verts_wing.shape(1);
    dims.vertex_stride_1 = verts_wing.stride(1);
    dims.vertex_stride_2 = verts_wing.stride(2);
    kernel_mac<<<grid_size(block, n)(), block()>>>(
        dims,
        verts_wing.ptr(),
        d_mac
    );
    CHECK_CUDA(cudaGetLastError());
    // stream sync
    f32 h_mac = 0.0f;
    memory->copy(Location::Host, &h_mac, 1, Location::Device, d_mac, 1, sizeof(f32), 1);
    return h_mac / sum(areas);
}

void BackendCUDA::gamma_wake_from_coeffs(
    TensorView2fD& gamma_wake,
    TensorView2fD& gamma_coeffs,
    i32 harmonics,
    f32 tn,
    f32 omega,
    f32 dt,
    i64 iteration
) {
    constexpr Dim3<i32> block{16, 32};
    Dim3<i64> n{gamma_wake.shape(0), iteration};
    auto gamma_wake_it = gamma_wake.slice(All, Range{-iteration, -1});
    DataDims dims;
    dims.panel_shape_0 = gamma_wake.shape(0);
    dims.panel_shape_1 = iteration;
    dims.panel_stride_1 = gamma_wake.stride(1);
    kernel_gamma_wake_from_coeffs<<<grid_size(block, n)(), block()>>>(
        dims, 
        gamma_wake_it.ptr(),
        gamma_coeffs.ptr(),
        gamma_coeffs.stride(1),
        harmonics,
        tn,
        omega,
        dt
    );
    CHECK_CUDA(cudaGetLastError());
}

f32 BackendCUDA::sum(TensorView1fD& tensor) {
    f32* d_sum = (f32*)memory->alloc(Location::Device, sizeof(*d_sum));
    f32 h_sum;
    cudaMemset(d_sum, 0, sizeof(f32)); // TODO: use fill instead ?
    constexpr Dim3<i32> block{768};
    Dim3<i64> n{tensor.size()};
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

f32 BackendCUDA::sum(TensorView2fD& tensor) {
    f32* d_sum = (f32*)memory->alloc(Location::Device, sizeof(*d_sum));
    f32 h_sum;
    cudaMemset(d_sum, 0, sizeof(f32));
    constexpr Dim3<i32> block{768};
    Dim3<i64> n{tensor.size()};
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
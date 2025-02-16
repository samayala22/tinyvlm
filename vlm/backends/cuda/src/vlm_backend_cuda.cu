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
            auto verts_wing_j = verts_wing[m_j];
            auto verts_wake_j = verts_wake[m_j];

            f32* lhs_section = lhs.ptr() + offset_i + offset_j * lhs.stride(1);

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
                const Dim3<i64> n{colloc_i.stride(2), colloc_j.shape(0)};
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
                const Dim3<i64> n{colloc_i.stride(2), colloc_j.shape(0)};
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
void BackendCUDA::rhs_assemble_velocities(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& velocities) {
    // const tiny::ScopedTimer timer("RHS");
    
    i64 offset = 0;
    for (i64 m = 0; m < normals.size(); m++) {
        assert(offset <= rhs.size());
        const auto& normals_i = normals[m];
        const auto& velocities_m = velocities[m];
        constexpr Dim3<i32> block{768};
        const Dim3<i64> n{normals_i.stride(2)};
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

void BackendCUDA::rhs_assemble_wake_influence(TensorView<f32, 1, Location::Device>& rhs, const MultiTensorView2D<Location::Device>& gamma_wake, const MultiTensorView3D<Location::Device>& colloc, const MultiTensorView3D<Location::Device>& normals, const MultiTensorView3D<Location::Device>& verts_wake, const std::vector<bool>& lifting, i32 iteration) {
    if (iteration == 0) return; // cuda doesnt support 0 sized domain
    constexpr Dim3<i32> block{32, 16}; // TODO: do not modify this 

    i64 offset = 0;
    for (i32 m_i = 0; m_i < normals.size(); m_i++) {
        const i64 m_i_num_panels = normals[m_i].shape(0) * normals[m_i].shape(1);
        for (i32 m_j = 0; m_j < normals.size(); m_j++) {
            if (!lifting[m_j]) continue;
            const auto& colloc_i = colloc[m_i];
            const auto& normals_i = normals[m_i];
            const auto& gamma_wake_j = gamma_wake[m_j];
            const auto& verts_wake_j = verts_wake[m_j];
            const i64 wake_m  = iteration; 
            const i64 wake_n  = verts_wake_j.shape(0) - 1; // spanwise number of wake panels
            const Dim3<i64> n{wake_m * wake_n, m_i_num_panels};
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

void BackendCUDA::displace_wake_rollup(MultiTensorView3D<Location::Device>& wake_rollup, const MultiTensorView3D<Location::Device>& verts_wake, const MultiTensorView3D<Location::Device>& verts_wing, const MultiTensorView2D<Location::Device>& gamma_wing, const MultiTensorView2D<Location::Device>& gamma_wake, f32 dt, i32 iteration) {
    // TODO
}

// TODO: deprecate
f32 BackendCUDA::coeff_steady_cl_single(const TensorView3D<Location::Device>& verts_wing, const TensorView2D<Location::Device>& gamma_delta, const FlowData& flow, f32 area) {
    f32 h_cl = 0.0f;
    cudaMemset(d_cl, 0, sizeof(f32));
    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{gamma_delta.shape(0), gamma_delta.shape(1)};
    kernel_coeff_steady_cl_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        n.y,
        n.x,
        verts_wing.ptr(),
        verts_wing.stride(1),
        verts_wing.stride(2),
        gamma_delta.ptr(),
        gamma_delta.stride(1),
        float3{flow.freestream.x, flow.freestream.y, flow.freestream.z},
        float3{flow.lift_axis.x, flow.lift_axis.y, flow.lift_axis.z},
        flow.rho,
        d_cl
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_cl, 1, Location::Device, d_cl, 1, sizeof(f32), 1);
    CtxManager::getInstance().sync();
    
    return h_cl / (0.5f * flow.rho * linalg::length2(flow.freestream) * area);
}

void BackendCUDA::forces_unsteady(
    const TensorView3D<Location::Device>& verts_wing,
    const TensorView2D<Location::Device>& gamma_delta,
    const TensorView2D<Location::Device>& gamma,
    const TensorView2D<Location::Device>& gamma_prev,
    const TensorView3D<Location::Device>& velocities,
    const TensorView2D<Location::Device>& areas,
    const TensorView3D<Location::Device>& normals,
    const TensorView3D<Location::Device>& forces,
    f32 dt
)
{
    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{gamma_delta.shape(0), gamma_delta.shape(1)};
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
        gamma.ptr(),
        gamma_prev.ptr(),
        velocities.ptr(),
        areas.ptr(),
        normals.ptr(),
        forces.ptr(),
        dt
    );
    CHECK_CUDA(cudaGetLastError());
    // stream sync
}

f32 BackendCUDA::coeff_cl(
    const TensorView3D<Location::Device>& forces,
    const linalg::float3& lift_axis,
    const linalg::float3& freestream,
    const f32 rho,
    const f32 area
)
{
    cudaMemset(d_cl, 0, sizeof(f32));
    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{forces.shape(0), forces.shape(1)};
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
    const TensorView3D<Location::Device>& forces,
    const TensorView3D<Location::Device>& verts_wing,
    const linalg::float3& ref_pt,
    const linalg::float3& freestream,
    const f32 rho,
    const f32 area,
    const f32 mac
)
{
    cudaMemset(d_cm_x, 0, sizeof(f32));
    cudaMemset(d_cm_y, 0, sizeof(f32));
    cudaMemset(d_cm_z, 0, sizeof(f32));

    constexpr Dim3<i32> block{32, 16};
    const Dim3<i64> n{forces.shape(0), forces.shape(1)};
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

f32 BackendCUDA::coeff_steady_cd_single(const TensorView3D<Location::Device>& verts_wake, const TensorView2D<Location::Device>& gamma_wake, const FlowData& flow, f32 area) {
    f32 h_cd = 0.0f;
    cudaMemset(d_cd, 0, sizeof(f32));
    constexpr Dim3<i32> block{64, 8};
    const Dim3<i64> n{verts_wake.shape(0)-1, verts_wake.shape(0)-1};
    kernel_coeff_steady_cd_single<block.x, block.y><<<grid_size(block, n)(), block()>>>(
        verts_wake.ptr(),
        verts_wake.stride(2),
        verts_wake.shape(1),
        verts_wake.shape(0),
        gamma_wake.ptr(),
        sigma_vatistas,
        d_cd
    );
    CHECK_CUDA(cudaGetLastError());
    memory->copy(Location::Host, &h_cd, 1, Location::Device, d_cd, 1, sizeof(f32), 1);
    CtxManager::getInstance().sync();
    return h_cd / (linalg::length2(flow.freestream) * area);
}

void BackendCUDA::mesh_metrics(const f32 alpha_rad, const MultiTensorView3D<Location::Device>& verts_wing, MultiTensorView3D<Location::Device>& colloc, MultiTensorView3D<Location::Device>& normals, MultiTensorView2D<Location::Device>& areas) {
    for (int m = 0; m < colloc.size(); m++) {
        const auto& colloc_i = colloc[m];
        const auto& normals_i = normals[m];
        const auto& areas_i = areas[m];
        const auto& verts_wing_m = verts_wing[m];

        constexpr Dim3<i32> block{32, 16}; // ns, nc
        const Dim3<i64> n{colloc_i.shape(0), colloc_i.shape(1)};
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
    const TensorView3D<Location::Device>& verts_wing,
    const TensorView2D<Location::Device>& areas
) 
{
    cudaMemset(d_mac, 0, sizeof(f32));
    constexpr Dim3<i32> block{768};
    const Dim3<i64> n{areas.shape(0)};
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
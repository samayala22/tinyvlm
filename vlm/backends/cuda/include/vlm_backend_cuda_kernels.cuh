#include "vlm_types.hpp"
#include "helper_math.cuh"
#include <cooperative_groups.h>
#include <cstdint>

namespace vlm {

namespace cg = cooperative_groups;

constexpr f32 RCUT = 1e-10f;
constexpr f32 RCUT2 = 1e-5f;

template<typename T = u32>
struct Dim3 {
    T x, y, z;
    __host__ __device__ constexpr Dim3(T x_, T y_= 1, T z_= 1) : x(x_), y(y_), z(z_) {}
    __host__ __device__ constexpr T size() const {return x * y * z;}
    __host__ __device__ constexpr dim3 operator()() const { return dim3(static_cast<u32>(x), static_cast<u32>(y), static_cast<u32>(z)); }
};

template<typename T>
constexpr Dim3<u32> grid_size(const Dim3<u32>& block, const Dim3<T>& size) {
    return Dim3{
        static_cast<u32>((size.x + block.x - 1) / block.x),
        static_cast<u32>((size.y + block.y - 1) / block.y),
        static_cast<u32>((size.z + block.z - 1) / block.z)
    };
}

template<Dim3 BlockSize>
__global__ void __launch_bounds__(BlockSize.size()) kernel_fill_f32(float* buffer, float value, size_t n) {
    const u64 tid = cg::this_grid().thread_rank();
    
    if (tid >= n) return;
    buffer[tid] = value;
}


// #define BlockSizeX 32
// #define BlockSizeY 16

__device__ inline float3 kernel_biosavart(float3 colloc, const float3 vertex1, const float3 vertex2, const float sigma) {
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

__device__ inline void kernel_symmetry(float3* inf, float3 colloc, const float3 vertex0, const float3 vertex1, const float sigma) {
    float3 induced_speed = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf->x += induced_speed.x;
    inf->y += induced_speed.y;
    inf->z += induced_speed.z;
    colloc.y = -colloc.y; // wing symmetry
    float3 induced_speed_sym = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf->x += induced_speed_sym.x;
    inf->y -= induced_speed_sym.y;
    inf->z += induced_speed_sym.z;
}

// start: starting linear index
// length: number of panels (columns) to process (from start to start+length)
// offset: offset between the linear index for the influenced panel and the influencing panel (used when influencing panel is part of the wake)
// Kernel achieves 32% of theoretical peak performance with 3.32 IPC and 85% of compute throughput


/// @param v ptr to a upper left vertex of a panel that is along the chord root
template<Dim3 BlockSize>
__global__ void __launch_bounds__(BlockSize.size()) kernel_influence(u64 m, u64 n, f32* lhs, f32* collocs, u64 collocs_ld, f32* v, u64 v_ld, u64 v_n, f32* normals, u64 normals_ld, f32 sigma) {
    assert(n % (v_n-1) == 0); // kernel runs on a span wise section of the surface
    const u64 j = blockIdx.y * blockDim.y + threadIdx.y;
    const u64 i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n || i >= m) return;

    __shared__ float sharedCollocX[BlockSize.x];
    __shared__ float sharedCollocY[BlockSize.x];
    __shared__ float sharedCollocZ[BlockSize.x];
    __shared__ float sharedNormalX[BlockSize.x];
    __shared__ float sharedNormalY[BlockSize.x];
    __shared__ float sharedNormalZ[BlockSize.x];

    // Load memory along warp onto the shared memory
    if (threadIdx.y == 0) {
        // Load colloc and normal data into shared memory
        sharedCollocX[threadIdx.x] = collocs[0*collocs_ld + i];
        sharedCollocY[threadIdx.x] = collocs[1*collocs_ld + i];
        sharedCollocZ[threadIdx.x] = collocs[2*collocs_ld + i];
        sharedNormalX[threadIdx.x] = normals[0*normals_ld + i];
        sharedNormalY[threadIdx.x] = normals[1*normals_ld + i];
        sharedNormalZ[threadIdx.x] = normals[2*normals_ld + i];
    }

    cg::this_thread_block().sync();

    float3 inf{0.0f, 0.0f, 0.0f};
    {
        const u64 v0 = j + j / (v_n-1);
        const u64 v1 = v0 + 1;
        const u64 v3 = v0 + v_n;
        const u64 v2 = v3 + 1;
        
        // Tried to put them in shared memory but got worse L1 hit rate. 
        const float3 vertex0{v[0*v_ld + v0], v[1*v_ld + v0], v[2*v_ld + v0]};
        const float3 vertex1{v[0*v_ld + v1], v[1*v_ld + v1], v[2*v_ld + v1]};
        const float3 vertex2{v[0*v_ld + v2], v[1*v_ld + v2], v[2*v_ld + v2]};
        const float3 vertex3{v[0*v_ld + v3], v[1*v_ld + v3], v[2*v_ld + v3]};

        // No bank conflicts as each thread reads a different index
        const float3 colloc = {sharedCollocX[threadIdx.x], sharedCollocY[threadIdx.x], sharedCollocZ[threadIdx.x]};

        kernel_symmetry(&inf, colloc, vertex0, vertex1, sigma);
        kernel_symmetry(&inf, colloc, vertex1, vertex2, sigma);
        kernel_symmetry(&inf, colloc, vertex2, vertex3, sigma);
        kernel_symmetry(&inf, colloc, vertex3, vertex0, sigma);
    }
    {
        const float3 normal = {sharedNormalX[threadIdx.x], sharedNormalY[threadIdx.x], sharedNormalZ[threadIdx.x]};
        lhs[j * m + i] += dot(inf, normal);
    }
}

template<Dim3 BlockSize>
__global__ void __launch_bounds__(BlockSize.size()) kernel_gamma_delta(u64 m, u64 n, f32* gamma_wing_delta, const f32* gamma_wing) {
    const u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    const u64 j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= n || i >= m) return;

    gamma_wing_delta[i * n + j] = gamma_wing[i * n + j] - gamma_wing[(i-1) * n + j];
}

}
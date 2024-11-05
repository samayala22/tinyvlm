#include "vlm_types.hpp"
#include "helper_math.cuh"
#include <cooperative_groups.h>
#include <cstdint>
#include <stdio.h> // printf debugging

namespace vlm {

namespace cg = cooperative_groups;

constexpr f32 RCUT = 1e-10f;
constexpr f32 RCUT2 = 1e-5f;

template<typename T = i32>
struct Dim3 {
    T x, y, z;
    __host__ __device__ constexpr Dim3(T x_, T y_= 1, T z_= 1) : x(x_), y(y_), z(z_) {}
    __host__ __device__ __inline__ constexpr T size() const {return x * y * z;}
    __host__ __device__ __inline__ dim3 operator()() const { return dim3(static_cast<i32>(x), static_cast<i32>(y), static_cast<i32>(z)); }
};

template<typename T>
constexpr Dim3<i32> grid_size(const Dim3<i32>& block, const Dim3<T>& size) {
    return Dim3{
        static_cast<i32>((size.x + block.x - 1) / block.x),
        static_cast<i32>((size.y + block.y - 1) / block.y),
        static_cast<i32>((size.z + block.z - 1) / block.z)
    };
}

template<typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    cg::coalesced_group active = cg::coalesced_threads();
    #pragma unroll
    for (i32 offset = active.size() / 2; offset > 0; offset /= 2) {
        val += active.shfl_down(val, offset);
    }
    return val;
}

template<typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    static __shared__ T shared[32]; // Shared mem for 32 partial sums
    i32 lane = threadIdx.x % warpSize;
    i32 wid = threadIdx.x / warpSize;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    val = warp_reduce_sum(val);     // reduce over warps of a block

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    block.sync();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = warp_reduce_sum(val); // Reduce the 32 sums of the block into a single value

    return val;
}

template<typename T>
__global__ void kernel_reduce(i64 N, T* buf, T* val) {
    T sum = 0;
    i64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop
    for (i32 i = tid; i < N; i += blockDim.x * gridDim.x) {
        sum += buf[i];
    }
    
    sum = block_reduce_sum(sum); // reduce over threads of a block
        
    if (threadIdx.x == 0) {
        atomicAdd(val, sum); // reduce over blocks
    }
}

template<typename T>
__global__ void kernel_fill(T* buffer, i64 stride, T value, i64 n) {
    const i64 tid = cg::this_grid().thread_rank();
    
    if (tid >= n) return;
    buffer[tid * stride] = value;
}

__inline__ __device__ float3 kernel_biosavart(float3 colloc, const float3 vertex1, const float3 vertex2, const float sigma) {
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

__inline__ __device__ void kernel_symmetry(float3* inf, float3 colloc, const float3 vertex0, const float3 vertex1, const float sigma) {
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
template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_influence(i64 m, i64 n, f32* lhs, f32* collocs, i64 collocs_ld, f32* v, i64 v_ld, i64 v_n, f32* normals, i64 normals_ld, f32 sigma) {
    assert(n % (v_n-1) == 0); // kernel runs on a span wise section of the surface
    const i64 j = blockIdx.y * blockDim.y + threadIdx.y;
    const i64 i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n || i >= m) return;

    __shared__ float sharedCollocX[X];
    __shared__ float sharedCollocY[X];
    __shared__ float sharedCollocZ[X];
    __shared__ float sharedNormalX[X];
    __shared__ float sharedNormalY[X];
    __shared__ float sharedNormalZ[X];

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
        const i64 v0 = j + j / (v_n-1);
        const i64 v1 = v0 + 1;
        const i64 v3 = v0 + v_n;
        const i64 v2 = v3 + 1;
        
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

template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_gamma_delta(i64 m, i64 n, f32* gamma_wing_delta, const f32* gamma_wing) {
    const i64 i = blockIdx.x * blockDim.x + threadIdx.x;
    const i64 j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= n || i >= m) return;

    gamma_wing_delta[i * n + j] = gamma_wing[i * n + j] - gamma_wing[(i-1) * n + j];
}

template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_rhs_assemble_velocities(i64 n, f32* rhs, const f32* velocities, i64 velocities_ld, const f32* normals, i64 normals_ld) {
    const i64 i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    rhs[i] += - (
        velocities[i + 0 * velocities_ld] * normals[i + 0 * normals_ld] +
        velocities[i + 1 * velocities_ld] * normals[i + 1 * normals_ld] +
        velocities[i + 2 * velocities_ld] * normals[i + 2 * normals_ld]
    );
}

template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_mesh_metrics(i64 m, i64 n, f32* colloc, i64 colloc_ld, f32* normals, i64 normals_ld, f32* areas, const f32* verts_wing, i64 verts_wing_ld, f32 alpha_rad) {
    const i64 j = blockIdx.x * blockDim.x + threadIdx.x;
    const i64 i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= n || i >= m) return;

    const i64 lidx =  i * n + j;
    const i64 v0 = (i+0) * (n+1) + j;
    const i64 v1 = (i+0) * (n+1) + j + 1;
    const i64 v2 = (i+1) * (n+1) + j + 1;
    const i64 v3 = (i+1) * (n+1) + j;

    const float3 vertex0{verts_wing[0*verts_wing_ld + v0], verts_wing[1*verts_wing_ld + v0], verts_wing[2*verts_wing_ld + v0]}; // upper left
    const float3 vertex1{verts_wing[0*verts_wing_ld + v1], verts_wing[1*verts_wing_ld + v1], verts_wing[2*verts_wing_ld + v1]}; // upper right
    const float3 vertex2{verts_wing[0*verts_wing_ld + v2], verts_wing[1*verts_wing_ld + v2], verts_wing[2*verts_wing_ld + v2]}; // lower right
    const float3 vertex3{verts_wing[0*verts_wing_ld + v3], verts_wing[1*verts_wing_ld + v3], verts_wing[2*verts_wing_ld + v3]}; // lower left

    const float3 normal_vec = normalize(cross(vertex3 - vertex1, vertex2 - vertex0));
    normals[0*normals_ld + lidx] = normal_vec.x;
    normals[1*normals_ld + lidx] = normal_vec.y;
    normals[2*normals_ld + lidx] = normal_vec.z;

    // 3 vectors f (P0P3), b (P0P2), e (P0P1) to compute the area:
    // area = 0.5 * (||f x b|| + ||b x e||)
    // this formula might also work: area = || 0.5 * ( f x b + b x e ) ||
    const float3 vec_f = vertex3 - vertex0;
    const float3 vec_b = vertex2 - vertex0;
    const float3 vec_e = vertex1 - vertex0;

    areas[lidx] = 0.5f * (length(cross(vec_f, vec_b)) + length(cross(vec_b, vec_e)));

    // High AoA correction (Aerodynamic Optimization of Aircraft Wings Using a Coupled VLM2.5D RANS Approach) Eq 3.4 p21
    // https://publications.polymtl.ca/2555/1/2017_MatthieuParenteau.pdf
    const f32 factor = (alpha_rad < EPS_f) ? 0.5f : 0.5f * (alpha_rad / (sin(alpha_rad) + EPS_f));
    const float3 chord_vec = 0.5f * (vertex2 + vertex3 - vertex0 - vertex1);
    const float3 colloc_pt = 0.5f * (vertex0 + vertex1) + factor * chord_vec;
    
    colloc[0*colloc_ld + lidx] = colloc_pt.x;
    colloc[1*colloc_ld + lidx] = colloc_pt.y;
    colloc[2*colloc_ld + lidx] = colloc_pt.z;
}

template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_coeff_steady_cl_single(i64 m, i64 n, const f32* verts_wing, i64 verts_wing_surface_ld, i64 verts_wing_ld, const f32* gamma_delta, i64 gamma_delta_surface_ld, float3 freestream, float3 lift_axis, f32 rho, f32* cl) {
    static_assert(X == 32);

    cg::thread_block block = cg::this_thread_block();

    const i64 j = blockIdx.x * blockDim.x + threadIdx.x;
    const i64 i = blockIdx.y * blockDim.y + threadIdx.y;
    const i32 wid = block.thread_rank() / warpSize;
    const i32 lane = block.thread_rank() % warpSize;

    __shared__ f32 shared[32];

    if (wid == 0) shared[lane] = 0.0f;
    block.sync();

    if (j >= n || i >= m) return;

    const i64 v0 = (i+0) * verts_wing_surface_ld + j;
    const i64 v1 = (i+0) * verts_wing_surface_ld + j + 1;
    const float3 vertex0{verts_wing[0*verts_wing_ld + v0], verts_wing[1*verts_wing_ld + v0], verts_wing[2*verts_wing_ld + v0]}; // upper left
    const float3 vertex1{verts_wing[0*verts_wing_ld + v1], verts_wing[1*verts_wing_ld + v1], verts_wing[2*verts_wing_ld + v1]}; // upper right
    // Leading edge vector pointing outward from wing root
    const float3 dl = vertex1 - vertex0;
    const float3 force = cross(freestream, dl) * rho * gamma_delta[i * gamma_delta_surface_ld + j];

    f32 cl_local = dot(force, lift_axis);

    atomicAdd(cl, cl_local); // naive reduction (works but 50x slower)
    // cl_local = warp_reduce_sum(cl_local); // reduce of the warp
    // if (lane == 0) shared[wid] = cl_local; // write to smem
    // block.sync(); // wait for threads to write to smem
    // if (wid == 0) cl_local = warp_reduce_sum(shared[lane]);// reduce of the block
    // if (block.thread_rank() == 0) atomicAdd(cl, cl_local); // reduce of the grid
}

inline __device__ float3 quad_normal(const float3 v0, const float3 v1, const float3 v2, const float3 v3) {
    return normalize(cross(v3-v1, v2-v0));
}

template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_coeff_steady_cd_single(f32* verts_wake, i64 verts_wake_ld, i64 verts_wake_m, i64 verts_wake_n, float* gamma_wake, f32 sigma, f32* cd) {
    cg::thread_block block = cg::this_thread_block();

    const i64 i = blockIdx.x * blockDim.x + threadIdx.x; // jj
    const i64 j = blockIdx.y * blockDim.y + threadIdx.y; // j
    const i32 wid = block.thread_rank() / warpSize;
    const i32 lane = block.thread_rank() % warpSize;

    __shared__ f32 shared[32];
    if (wid == 0) shared[lane] = 0.0f;
    block.sync();
 
    if (i >= verts_wake_n - 1 || j >= verts_wake_n - 1) return;
    
    const i64 vi = verts_wake_m - 2;

    const i64 v0 = (vi+0) * (verts_wake_n) + j;
    const i64 v1 = (vi+0) * (verts_wake_n) + j + 1;
    const i64 v2 = (vi+1) * (verts_wake_n) + j + 1;
    const i64 v3 = (vi+1) * (verts_wake_n) + j;

    const float3 vertex0 = {verts_wake[0*verts_wake_ld + v0], verts_wake[1*verts_wake_ld + v0], verts_wake[2*verts_wake_ld + v0]};
    const float3 vertex1 = {verts_wake[0*verts_wake_ld + v1], verts_wake[1*verts_wake_ld + v1], verts_wake[2*verts_wake_ld + v1]};
    const float3 vertex2 = {verts_wake[0*verts_wake_ld + v2], verts_wake[1*verts_wake_ld + v2], verts_wake[2*verts_wake_ld + v2]};
    const float3 vertex3 = {verts_wake[0*verts_wake_ld + v3], verts_wake[1*verts_wake_ld + v3], verts_wake[2*verts_wake_ld + v3]};

    const float3 colloc = 0.25f * (vertex0 + vertex1 + vertex2 + vertex3); // 3*(3 add + 1 mul)
    const float3 normal = quad_normal(vertex0, vertex1, vertex2, vertex3);

    const i64 vv0 = (vi+0) * (verts_wake_n) + i;
    const i64 vv1 = (vi+0) * (verts_wake_n) + i + 1;
    const i64 vv2 = (vi+1) * (verts_wake_n) + i + 1;
    const i64 vv3 = (vi+1) * (verts_wake_n) + i;

    const float3 vvertex0 = {verts_wake[0*verts_wake_ld + vv0], verts_wake[1*verts_wake_ld + vv0], verts_wake[2*verts_wake_ld + vv0]};
    const float3 vvertex1 = {verts_wake[0*verts_wake_ld + vv1], verts_wake[1*verts_wake_ld + vv1], verts_wake[2*verts_wake_ld + vv1]};
    const float3 vvertex2 = {verts_wake[0*verts_wake_ld + vv2], verts_wake[1*verts_wake_ld + vv2], verts_wake[2*verts_wake_ld + vv2]};
    const float3 vvertex3 = {verts_wake[0*verts_wake_ld + vv3], verts_wake[1*verts_wake_ld + vv3], verts_wake[2*verts_wake_ld + vv3]};

    float3 inf = {0.0f, 0.0f, 0.0f};

    kernel_symmetry(&inf, colloc, vvertex1, vvertex2, sigma);
    kernel_symmetry(&inf, colloc, vvertex3, vvertex0, sigma);

    const float gammaw = gamma_wake[vi * (verts_wake_n-1) + i];
    const f32 cd_local = - gamma_wake[vi * (verts_wake_n-1) + j] * dot(gammaw * inf, normal) * length(vertex1 - vertex0);
    
    atomicAdd(cd, cd_local);
    // cd_local = warp_reduce_sum(cd_local);
    // if (lane == 0) shared[wid] = cd_local;
    // block.sync();
    // if (wid == 0) cd_local = warp_reduce_sum(shared[lane]);
    // if (block.thread_rank() == 0) atomicAdd(cd, cd_local);
}

template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_wake_influence(i64 wake_m, i64 wake_n, i64 wing_mn, const f32* colloc, const i64 colloc_ld, const f32* normals, const i64 normals_ld, const f32* verts_wake, const i64 verts_wake_ld, const f32* gamma_wake, f32* rhs, f32 sigma) {
    static_assert(X == 32);
    const cg::thread_block block = cg::this_thread_block();
    const i64 i = blockIdx.x * blockDim.x + threadIdx.x; // wake_verts
    const i64 j = blockIdx.y * blockDim.y + threadIdx.y; // colloc
    const i32 wid = block.thread_rank() / warpSize;
    const i32 lane = block.thread_rank() % warpSize;
    
    float induced_vel = 0.0f;
    if (i < wake_m * wake_n || j < wing_mn) {
        const i64 v0 = i + i / wake_n;
        const i64 v1 = v0 + 1;
        const i64 v3 = v0 + wake_n+1;
        const i64 v2 = v3 + 1;

        const float3 colloc_influenced = {colloc[0*colloc_ld + j], colloc[1*colloc_ld + j], colloc[2*colloc_ld + j]};
        const float3 normal = {normals[0*normals_ld + j], normals[1*normals_ld + j], normals[2*normals_ld + j]};

        const float3 vertex0 = {verts_wake[0*verts_wake_ld + v0], verts_wake[1*verts_wake_ld + v0], verts_wake[2*verts_wake_ld + v0]};
        const float3 vertex1 = {verts_wake[0*verts_wake_ld + v1], verts_wake[1*verts_wake_ld + v1], verts_wake[2*verts_wake_ld + v1]};
        const float3 vertex2 = {verts_wake[0*verts_wake_ld + v2], verts_wake[1*verts_wake_ld + v2], verts_wake[2*verts_wake_ld + v2]};
        const float3 vertex3 = {verts_wake[0*verts_wake_ld + v3], verts_wake[1*verts_wake_ld + v3], verts_wake[2*verts_wake_ld + v3]};

        float3 ind = {0.0f, 0.0f, 0.0f};

        kernel_symmetry(&ind, colloc_influenced, vertex0, vertex1, sigma);
        kernel_symmetry(&ind, colloc_influenced, vertex1, vertex2, sigma);
        kernel_symmetry(&ind, colloc_influenced, vertex2, vertex3, sigma);
        kernel_symmetry(&ind, colloc_influenced, vertex3, vertex0, sigma);

        induced_vel = dot(ind * gamma_wake[i], normal);
    }

    // atomicAdd(rhs + j, -induced_vel); // naive reduction
    induced_vel = warp_reduce_sum(induced_vel); // Y warp reductions

    if (threadIdx.x == 0) {
        atomicAdd(rhs + j, -induced_vel);
    }
}

template<i32 X, i32 Y = 1, i32 Z = 1>
__global__ void __launch_bounds__(X*Y*Z) kernel_coeff_unsteady_cl_single(i64 m, i64 n, i64 ld, const f32* verts_wing, i64 verts_wing_ld, const f32* gamma_wing_delta, const f32* gamma_wing, const f32* gamma_wing_prev, const f32* velocities, i64 velocities_ld, const f32* areas, const f32* normals, i64 normals_ld, float3 freestream, f32 dt, f32* cl) {
    const cg::thread_block block = cg::this_thread_block();
    const i64 j = blockIdx.x * blockDim.x + threadIdx.x; // wake_verts
    const i64 i = blockIdx.y * blockDim.y + threadIdx.y; // colloc
    const i32 wid = block.thread_rank() / warpSize;
    const i32 lane = block.thread_rank() % warpSize;

    if (i >= m|| j >= n) return;
    
    const f32 rho = 1.0f; // TODO: remove hardcoded rho
    const float3 span_axis{0.f, 1.f, 0.f}; // TODO: obtain from the local frame
    const float3 lift_axis = normalize(cross(freestream, span_axis));

    const i64 lidx = i * ld + j;

    const float3 vel{velocities[0*velocities_ld + lidx], velocities[1*velocities_ld + lidx], velocities[2*velocities_ld + lidx]};

    const i64 v0 = (i+0) * (ld + 1) + j;
    const i64 v1 = (i+0) * (ld + 1) + j + 1;

    const float3 vertex0{verts_wing[0*verts_wing_ld + v0], verts_wing[1*verts_wing_ld + v0], verts_wing[2*verts_wing_ld + v0]}; // upper left
    const float3 vertex1{verts_wing[0*verts_wing_ld + v1], verts_wing[1*verts_wing_ld + v1], verts_wing[2*verts_wing_ld + v1]}; // upper right
    const float3 normal{normals[0*normals_ld + lidx], normals[1*normals_ld + lidx], normals[2*normals_ld + lidx]};
    
    float3 force{0.f, 0.f, 0.f};
    const f32 gamma_dt = (gamma_wing[lidx] - gamma_wing_prev[lidx]) / dt; // backward difference
    
    // Joukowski method
    force += rho * gamma_wing_delta[lidx] * cross(vel, vertex1 - vertex0); // steady contribution
    force += rho * gamma_dt * areas[lidx] * normal; // unsteady contribution
    float cl_local = dot(force, lift_axis);

    atomicAdd(cl, cl_local);
    // cl_local = warp_reduce_sum(cl_local);
    // if (lane == 0) {
    //     atomicAdd(cl, cl_local);
    // }
}
} // namespace vlm
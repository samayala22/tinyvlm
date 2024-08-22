#include "vlm_types.hpp"
#include <cooperative_groups.h>
#include <cstdint>

namespace vlm {

namespace cg = cooperative_groups;

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
        size_t tid = cg::this_grid().thread_rank();
        
        if (tid >= n) return;
        buffer[tid] = value;
    }

}
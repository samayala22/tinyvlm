#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <vector>
#include <array>
#include <memory>

#include "vlm_types.hpp"

namespace vlm {

enum class MemoryLocation {
    Device, Host, HostDevice
};

enum class MemoryTransfer {
    HostToDevice, DeviceToHost, HostToHost, DeviceToDevice
};

class Memory {
    public:
        Memory(bool unified) : _unified(unified) {}
        ~Memory() = default;
        virtual void* alloc(MemoryLocation location, std::size_t size) const = 0;
        virtual void free(MemoryLocation location, void* ptr) const = 0;
        virtual void copy(MemoryTransfer transfer, void* dst, const void* src, std::size_t size) const = 0;
        virtual void fill_f32(MemoryLocation location, float* ptr, float value, std::size_t size) const = 0;
        bool is_unified() const { return _unified; }
    private:
        const bool _unified; // host and device memory are the same
};

template<typename T>
struct LayoutTraits {
    static_assert(std::is_same_v<decltype(std::declval<T>().size()), std::size_t>,
                  "Layout must have a size() method returning std::size_t");
};

template<typename T, class Layout>
class View {
    static_assert(std::is_same_v<decltype(LayoutTraits<Layout>{}), LayoutTraits<Layout>>, "Incorrect Layout");

    public:
        T* ptr = nullptr; // required
        Layout layout;

        View() = default;
        View(T* ptr, const Layout& layout) : ptr(ptr), layout(layout) {}
        ~View() = default;

        std::size_t size() const { return layout.size(); }
        std::size_t size_bytes() const { return layout.size() * sizeof(T); }
        inline T& operator[](std::size_t index) {
            assert(ptr != nullptr && "Pointer is null");
            // assert(index < layout.size() && "Index out of bounds");
            return ptr[index];
        }
        inline const T& operator[](std::size_t index) const {
            assert(ptr != nullptr && "Pointer is null");
            // assert(index < layout.size() && "Index out of bounds");
            return ptr[index];
        }
};

template<typename T, MemoryLocation Location, class Layout>
class Buffer {
    using is_host = std::bool_constant<Location == MemoryLocation::Host || Location == MemoryLocation::HostDevice>;
    using is_device = std::bool_constant<Location == MemoryLocation::Device || Location == MemoryLocation::HostDevice>;

public:
    Buffer() = delete;
    explicit Buffer(const Memory& memory, const Layout& layout) : memory(memory) { alloc(layout); }
    explicit Buffer(const Memory& memory) : memory(memory) {}
    ~Buffer() { dealloc(); }

    View<T, Layout>& h_view() {
        static_assert(is_host::value);
        return _host;
    }
    View<T, Layout>& d_view() {
        static_assert(is_device::value);
        return _device;
    }

    void to_device() {
        static_assert(is_host::value && is_device::value);
        if (!memory.is_unified())
            memory.copy(MemoryTransfer::HostToDevice, _device.ptr, _host.ptr, size_bytes());
    }

    void to_host() {
        static_assert(is_host::value && is_device::value);
        if (!memory.is_unified()) 
            memory.copy(MemoryTransfer::DeviceToHost, _host.ptr, _device.ptr, size_bytes());
    }

    std::size_t size() const { return _host.size(); }
    std::size_t size_bytes() const { return _host.size() * sizeof(T); }

    void alloc(const Layout& layout) {
        _host.layout = layout;
        _device.layout = layout;

        if (memory.is_unified()) {
            _host.ptr = static_cast<T*>(memory.alloc(MemoryLocation::Host, size_bytes()));
            _device.ptr = _host.ptr;
        } else {
            if constexpr (is_host::value) _host.ptr = static_cast<T*>(memory.alloc(MemoryLocation::Host, size_bytes()));
            if constexpr (is_device::value) _device.ptr = static_cast<T*>(memory.alloc(MemoryLocation::Device, size_bytes()));
        }
    }

    void dealloc() {
        if (memory.is_unified()) {
            memory.free(MemoryLocation::Host, _host.ptr);
        } else {
            if constexpr (is_host::value) memory.free(MemoryLocation::Host, _host.ptr);
            if constexpr (is_device::value) memory.free(MemoryLocation::Device, _device.ptr);
        }
    }

    const Memory& memory;

private:
    View<T, Layout> _host;
    View<T, Layout> _device;

    Buffer(Buffer&&) = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct SurfaceDims { // ns major
    uint64_t nc; // number of chordwise elements
    uint64_t ns; // number of spanwise elements
    uint64_t offset; // offset of the surface in the buffer
    uint64_t size() const {return nc * ns;}
};

class SingleSurface {
    public:

        SingleSurface() = default;
        SingleSurface(uint64_t nc, uint64_t ns, uint64_t ld, uint64_t stride, uint32_t dim) {
            _nc = nc;
            _ns = ns;
            _ld = ld;
            _stride = stride;
            _dim = dim;
        }
        ~SingleSurface() = default;

        std::size_t size() const {return _ns * _nc; } // required
        uint64_t nc() const {return _nc; }
        uint64_t ns() const {return _ns; }
        uint64_t ld() const {return _ld; }
        uint64_t stride() const {return _stride; }
        uint64_t dim() const {return _dim; }

    private:
        uint64_t _nc = 0; // chord wise
        uint64_t _ns = 0; // span wise
        uint64_t _ld = 0; // distance between two consecutive rows 
        uint64_t _stride = 0; // distance between each dimension
        uint32_t _dim = 1;
};

class MultiSurface {
    public:

        MultiSurface() = default;
        MultiSurface(const std::vector<SurfaceDims>& surfaces, uint32_t dim) { construct(surfaces, dim); }
        ~MultiSurface() = default;

        void construct(const std::vector<SurfaceDims>& surfaces, uint32_t dim) {
            _surfaces = surfaces;
            _stride = surfaces.back().offset + surfaces.back().size();
            _dim = dim;
        }

        uint64_t operator()(uint32_t wing_id, uint32_t dim) { // returns start of the specific wing buffer
            assert(wing_id < _surfaces.size());
            assert(dim < _dim);
            return _surfaces[wing_id].offset + dim * _stride;
        }

        std::size_t size() const {return dims() * stride(); } // required
        const std::vector<SurfaceDims>& surfaces() const {return _surfaces; }
        const SurfaceDims& surface(uint32_t wing_id) const {return _surfaces[wing_id]; }
        uint64_t stride() const {return _stride; }
        uint32_t dims() const {return _dim; }

        uint64_t nc(uint32_t wing_id) const {return _surfaces[wing_id].nc; }
        uint64_t ns(uint32_t wing_id) const {return _surfaces[wing_id].ns; }
        uint64_t offset(uint32_t wing_id) const {return _surfaces[wing_id].offset; }
        
        template<typename T>
        View<T, SingleSurface> subview(T* ptr, uint32_t wing_id) const {
            return subview(ptr, wing_id, 0, nc(wing_id), 0, ns(wing_id));
        }
        
        template<typename T>
        View<T, SingleSurface> subview(T* ptr, uint32_t wing_id, uint64_t i, uint64_t m, uint64_t j, uint64_t n) const {
            assert(wing_id < _surfaces.size());
            assert(i + m <= nc(wing_id));
            assert(j + n <= ns(wing_id));

            return {
                ptr + offset(wing_id) + i * ns(wing_id) + j,
                SingleSurface{m, n, ns(wing_id), stride(), dims()}
            };
        }

    private:
        std::vector<SurfaceDims> _surfaces;
        uint64_t _stride = 0;
        uint32_t _dim = 1;
};

enum class MatrixLayout {
    RowMajor, ColMajor
};

template<MatrixLayout Layout>
class Matrix {
    public:
        Matrix() = default;
        Matrix(uint64_t m, uint64_t n, uint64_t stride) { construct(m, n, stride); }
        void construct(uint64_t m, uint64_t n, uint64_t stride) {
            _m = m;
            _n = n;
            _stride = stride;
        }

        std::size_t size() const {return _m * _n; }
        uint64_t m() const {return _m; }
        uint64_t n() const {return _n; }
        uint64_t stride() const {return _stride; }
        constexpr MatrixLayout layout() const {return Layout; }

    private:
        uint64_t _m = 0;
        uint64_t _n = 0;
        uint64_t _stride = 0; // leading dimension
};

using Range = std::array<int, 2>;

constexpr Range all{0,-1};

template<typename... Args>
struct CountRanges;

template<>
struct CountRanges<> {
    static constexpr std::size_t value = 0;
};

template<typename First, typename... Rest>
struct CountRanges<First, Rest...> {
    static constexpr std::size_t value = std::is_same<Range, std::decay_t<First>>::value + CountRanges<Rest...>::value;
};

template<int Dim>
class Tensor {
    public:
        constexpr Tensor() = default;
        constexpr Tensor(const std::array<uint64_t, Dim>& shape, const std::array<uint64_t, Dim>& strides) : _shape(shape), _strides(strides) {}
        constexpr Tensor(const std::array<uint64_t, Dim>& shape) : _shape(shape) { default_strides(); }
        std::size_t size() const {
            std::size_t size = 1;
            for (std::size_t i = 0; i < Dim; i++) size *= _shape[i];
            return size;
        }

        uint64_t stride(int dim) const {return _strides[dim];}
        uint64_t shape(int dim) const {return _shape[dim];}

        template<typename T, typename... Args>
        inline constexpr auto slice(T* ptr, Args... args) {
            constexpr uint64_t M = CountRanges<Args...>::value;
            static_assert(sizeof...(args) == Dim, "The number of indices must match the dimension N.");
            static_assert(M <= Dim, "Too many ranges provided compared to the view's dimensionality");

            T* newPtr = ptr;
            std::array<uint64_t, M> newDims{};
            std::array<uint64_t, M> newStrides{};
            uint64_t newDimIndex = 0;

            uint64_t argIndex = 0;
            ([&](auto& arg) {
                if constexpr (std::is_same<std::decay_t<decltype(arg)>, Range>::value) {
                    uint64_t first = arg[0]; // Removed 'constexpr'
                    uint64_t last = (arg[1] < 0) ? _shape[argIndex] + arg[1] + 1 : arg[1];
                    assert((first >= 0) && (first < _shape[argIndex]));
                    assert((last >= 0) && (last <= _shape[argIndex])); // Changed '<' to '<=' for upper bound
                    assert(last - first > 0);
                    newPtr += first * _strides[argIndex];
                    newDims[newDimIndex] = last - first;
                    newStrides[newDimIndex] = _strides[argIndex];
                    newDimIndex++;
                } else if constexpr (std::is_integral<std::decay_t<decltype(arg)>>::value) {
                    uint64_t real_arg = (arg < 0) ? _shape[argIndex] + arg : arg; // Removed '+1' as indexing starts from 0
                    assert((real_arg >= 0) && (real_arg < _shape[argIndex]));
                    newPtr += real_arg * _strides[argIndex];
                }
                argIndex++;
            }(args), ...);

            return View<T, Tensor<M>>(newPtr, Tensor<M>{newDims, newStrides});
        }
    private:
        constexpr void default_strides() {
            _strides[0] = 1;
            for (std::size_t i = 1; i < Dim; ++i) {
                _strides[i] = _strides[i - 1] * _shape[i - 1];
            }
        }
        std::array<uint64_t, Dim> _shape{0};
        std::array<uint64_t, Dim> _strides{0};
};

enum class MemoryLocationV2 {
    Device, Host
};

// template<typename T, int Dim>
// class TensorView {
//     using DimArray = std::array<u64, Dim>;
//     public:
//         TensorView(const Memory& memory, MemoryLocationV2 location, const DimArray& shape, const DimArray& strides, T* ptr) : m_memory(memory), m_location(location), m_shape(shape), m_strides(strides), m_ptr(ptr) {}
//         ~TensorView() = default;

//         T* ptr() const {return m_ptr;}
        
//         u64 size() const {
//             u64 size = 1;
//             for (u64 i = 0; i < Dim; i++) size *= m_shape[i];
//             return size;
//         }

//         u64 stride(int dim) const {return m_strides[dim];}
//         u64 shape(int dim) const {return m_shape[dim];}

//         template<typename... Args>
//         inline constexpr auto slice(Args... args) {
//             constexpr u64 M = CountRanges<Args...>::value;
//             static_assert(sizeof...(args) == Dim, "The number of indices must match the dimension N.");
//             static_assert(M <= Dim, "Too many ranges provided compared to the view's dimensionality");

//             T* newPtr = m_ptr;
//             std::array<uint64_t, M> new_shape{};
//             std::array<uint64_t, M> newStrides{};
//             uint64_t newDimIndex = 0;

//             uint64_t argIndex = 0;
//             ([&](auto& arg) {
//                 if constexpr (std::is_same<std::decay_t<decltype(arg)>, Range>::value) {
//                     uint64_t first = arg[0]; // Removed 'constexpr'
//                     uint64_t last = (arg[1] < 0) ? m_shape[argIndex] + arg[1] + 1 : arg[1];
//                     assert((first >= 0) && (first < m_shape[argIndex]));
//                     assert((last >= 0) && (last <= m_shape[argIndex])); // Changed '<' to '<=' for upper bound
//                     assert(last - first > 0);
//                     newPtr += first * m_strides[argIndex];
//                     new_shape[newDimIndex] = last - first;
//                     newStrides[newDimIndex] = m_strides[argIndex];
//                     newDimIndex++;
//                 } else if constexpr (std::is_integral<std::decay_t<decltype(arg)>>::value) {
//                     uint64_t real_arg = (arg < 0) ? m_shape[argIndex] + arg : arg; // Removed '+1' as indexing starts from 0
//                     assert((real_arg >= 0) && (real_arg < m_shape[argIndex]));
//                     newPtr += real_arg * m_strides[argIndex];
//                 }
//                 argIndex++;
//             }(args), ...);
            
//             return TensorView<T, M>(m_memory, m_location, new_shape, newStrides, newPtr);
//         }
//     private:
//         const Memory& m_memory;
//         MemoryLocationV2 m_location;
//         DimArray m_shape{0};
//         DimArray m_strides{0};
//         T* m_ptr = nullptr;
// };

// template<typename T, int Dim>
// class TensorV2 : public TensorView<T, Dim> {
//     public:
//     TensorV2(const Memory& memory) : m_memory(memory) {}
//     void init(const DimArray& shape) {
//         m_shape = shape;
//         m_strides[0] = 1;
//         for (u64 i = 1; i < Dim; ++i) {
//             m_strides[i] = m_strides[i - 1] * m_shape[i - 1];
//         }
//     }

//     void init(const DimArray& shape, const DimArray& strides) {
//         m_shape = shape;
//         m_strides = strides;
//     }
// };

} // namespace vlm

// #include <algorithm>
// #include <cstring>
// #include <memory>

// using namespace vlm;

// class MemoryCPU final : public Memory {
//     public:
//         MemoryCPU() : Memory(true) {}
//         ~MemoryCPU() = default;
//         void* alloc(MemoryLocation location, std::size_t size) const override {return std::malloc(size);}
//         void free(MemoryLocation location, void* ptr) const override {std::free(ptr);}
//         void copy(MemoryTransfer transfer, void* dst, const void* src, std::size_t size) const override {std::memcpy(dst, src, size);}
//         void fill_f32(MemoryLocation location, float* ptr, float value, std::size_t size) const override {std::fill(ptr, ptr + size, value);}
// };

// int main() {
//     std::unique_ptr<Memory> memory = std::make_unique<MemoryCPU>(); // memory manager
//     Buffer<float, MemoryLocation::Host, MultiSurface> buffer{*memory}; // owning host/device buffer with a view type

//     MultiSurface layout{{{5,5,0}},3}; // dimensions of the view following the custom layout

//     buffer.alloc(layout); // allocate and initialize the views in the buffer.

//     assert(buffer.size() == layout.size()); // linear size of the allocated buffer must match the size of the view
// }
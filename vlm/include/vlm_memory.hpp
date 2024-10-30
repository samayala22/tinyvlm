#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <vector>
#include <array>
#include <memory>
#include <functional>

#include "vlm_types.hpp"

namespace vlm {

// TODO: remove HostDevice
enum class Location {
    Device, Host, HostDevice
};

class Memory {
    public:
        Memory(bool unified) : m_unified(unified) {}
        virtual ~Memory() = default;
        virtual void* alloc(Location location, i64 size) const = 0;
        virtual void free(Location location, void* ptr) const = 0;
        virtual void copy(Location dst_loc, void* dst, i64 dst_stride, Location src_loc, const void* src, i64 src_stride, i64 elem_size, i64 size) const = 0;
        virtual void fill(Location loc, float* ptr, float value, i64 size) const = 0;
        virtual void fill(Location loc, double* ptr, double value, i64 size) const = 0;

        bool is_unified() const { return m_unified; }
    private:
        const bool m_unified;
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

template<typename T, Location Location, class Layout>
class Buffer {
    using is_host = std::bool_constant<Location == Location::Host || Location == Location::HostDevice>;
    using is_device = std::bool_constant<Location == Location::Device || Location == Location::HostDevice>;

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
            memory.copy(Location::Device, _device.ptr, 1, Location::Host, _host.ptr, 1, sizeof(T), size());
    }

    void to_host() {
        static_assert(is_host::value && is_device::value);
        if (!memory.is_unified()) 
            memory.copy(Location::Host, _host.ptr, 1, Location::Device, _device.ptr, 1, sizeof(T), size());
    }

    std::size_t size() const { return _host.size(); }
    std::size_t size_bytes() const { return _host.size() * sizeof(T); }

    void alloc(const Layout& layout) {
        _host.layout = layout;
        _device.layout = layout;

        if (memory.is_unified()) {
            _host.ptr = static_cast<T*>(memory.alloc(Location::Host, size_bytes()));
            _device.ptr = _host.ptr;
        } else {
            if constexpr (is_host::value) _host.ptr = static_cast<T*>(memory.alloc(Location::Host, size_bytes()));
            if constexpr (is_device::value) _device.ptr = static_cast<T*>(memory.alloc(Location::Device, size_bytes()));
        }
    }

    void dealloc() {
        if (memory.is_unified()) {
            memory.free(Location::Host, _host.ptr);
        } else {
            if constexpr (is_host::value) memory.free(Location::Host, _host.ptr);
            if constexpr (is_device::value) memory.free(Location::Device, _device.ptr);
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

namespace beta {

using Range = std::array<i64, 2>;

constexpr Range All{0,-1};

template<typename... Args>
struct CountRanges;

template<>
struct CountRanges<> {
    static constexpr i64 value = 0;
};

template<typename First, typename... Rest>
struct CountRanges<First, Rest...> {
    static constexpr i64 value = std::is_same<Range, std::decay_t<First>>::value + CountRanges<Rest...>::value;
};

template<typename T, int Dim, Location L>
class TensorView {
    using DimArray = std::array<i64, Dim>;
    public:
        TensorView(const Memory& memory, const DimArray& shape, const DimArray& stride, T* const ptr) : m_memory(memory), m_shape(shape), m_stride(stride), m_ptr(ptr) {}
        ~TensorView() = default;

        inline i64 size() const {
            i64 size = 1;
            for (i64 i = 0; i < Dim; i++) size *= m_shape[i];
            return size;
        }

        inline i64 size_bytes() const { return size() * sizeof(T); }
        inline const DimArray& stride() const { return m_stride; }
        inline i64 stride(u32 dim) const { assert(dim < Dim); return m_stride[dim]; }
        inline const DimArray& shape() const { return m_shape; }
        inline i64 shape(u32 dim) const { assert(dim < Dim); return m_shape[dim]; }
        inline constexpr u32 dim() const { return Dim;}
        inline constexpr Location location() const { return L; }
        inline T* ptr() const {return m_ptr;}

        template<typename... Idx> inline constexpr T& operator()(Idx... idx)       { static_assert(L == Location::Host); return m_ptr[offset({idx...})]; }
        template<typename... Idx> inline constexpr T& operator()(Idx... idx) const { static_assert(L == Location::Host); return m_ptr[offset({idx...})]; }

        inline constexpr T& operator()(const DimArray& indices)       { static_assert(L == Location::Host); return m_ptr[offset(indices)]; }
        inline constexpr T& operator()(const DimArray& indices) const { static_assert(L == Location::Host); return m_ptr[offset(indices)]; }

        template<typename... Args>
        inline constexpr auto slice(Args... args) {
            constexpr i64 M = CountRanges<Args...>::value;
            static_assert(sizeof...(args) == Dim, "The number of indices must match the dimension N.");
            static_assert(M <= Dim, "Too many ranges provided compared to the view's dimensionality");

            T* newPtr = m_ptr;
            std::array<i64, M> new_shape{};
            std::array<i64, M> newstride{};
            i64 newDimIndex = 0;

            i64 argIndex = 0;
            ([&](auto& arg) {
                if constexpr (std::is_same<std::decay_t<decltype(arg)>, Range>::value) {
                    i64 first = arg[0];
                    i64 last = (arg[1] < 0) ? m_shape[argIndex] + arg[1] + 1 : arg[1];
                    assert((first >= 0) && (first < m_shape[argIndex]));
                    assert((last >= 0) && (last <= m_shape[argIndex]));
                    assert(last - first > 0);
                    newPtr += first * m_stride[argIndex];
                    new_shape[newDimIndex] = last - first;
                    newstride[newDimIndex] = m_stride[argIndex];
                    newDimIndex++;
                } else if constexpr (std::is_integral<std::decay_t<decltype(arg)>>::value) {
                    i64 real_arg = (arg < 0) ? m_shape[argIndex] + arg : arg;
                    assert((real_arg >= 0) && (real_arg < m_shape[argIndex]));
                    newPtr += real_arg * m_stride[argIndex];
                }
                argIndex++;
            }(args), ...);
            
            return TensorView<T, M, L>(m_memory, new_shape, newstride, newPtr);
        }

        template<Location ML>
        void to(TensorView<T, Dim, ML>& dst) const {
            assert(dst.shape() == m_shape);

            // TODO: Check for overlap
            // I believe this operation is too expensive and has a O(n^2) complexity

            // Compute number of consecutive constant strided elements
            i64 contiguous = shape(0);
            i64 i =  1;
            for (; i < Dim; i++) {
                if (stride(i)/stride(0) != contiguous || dst.stride(i)/dst.stride(0) != contiguous) break; // can put this in for loop body
                contiguous *= shape(i);
            }

            DimArray dim_idx{0};

            std::function<void(u32)> copy_lambda = [&](u32 di) {
                if (di == i-1) {
                    m_memory.copy(
                        dst.location(),
                        dst.ptr() + dst.offset(dim_idx),
                        dst.stride(0),
                        this->location(),
                        this->ptr() + this->offset(dim_idx),
                        this->stride(0),
                        sizeof(T),
                        contiguous
                    );
                } else {
                    for (u32 dii = 0; dii < shape(di); dii++) {
                        dim_idx[di] = dii;
                        copy_lambda(di-1);
                    }
                }
            };
            copy_lambda(Dim-1);
        }

        inline T& operator[](i64 i) { static_assert(L == Location::Host); return ptr()[i]; }

        inline constexpr i64 offset(const DimArray& indices) const {
            i64 index = 0;
            for (i64 i = 0; i < Dim; i++) {
                i64 idx = (indices[i] < 0) ? indices[i] + m_shape[i] : indices[i];
                index += idx * m_stride[i];
            }
            return index;
        }

        const Memory& m_memory;
    private:

        const DimArray m_shape{0};
        const DimArray m_stride{0};
        T* const m_ptr = nullptr;
};

template<typename T, int Dim, Location L>
class Tensor {
private:
    using View = TensorView<T, Dim, L>;
    using DimArray = std::array<i64, Dim>;
    View m_view;
    
public:
    Tensor(const Memory& memory) : 
        m_view(memory, {}, {}, nullptr) {}
    
    ~Tensor() {
        m_view.m_memory.free(L, m_view.ptr());
    }

    void init(const DimArray& shape) {
        auto& mem = m_view.m_memory;
        mem.free(L, m_view.ptr());
        i64 size = shape[0];
        DimArray stride;
        stride[0] = 1;
        for (i64 i = 1; i < Dim; ++i) {
            size *= shape[i];
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        T* const ptr = static_cast<T*>(mem.alloc(L, size * sizeof(T)));

        m_view.~View();
        new (&m_view) View(mem, shape, stride, ptr);
    }

    [[nodiscard]] Tensor<T, Dim, L> clone() {
        Tensor<T, Dim, L> cloned_tensor(m_view.m_memory);
        cloned_tensor.init(m_view.shape());
        m_view.to(cloned_tensor.view());
        return cloned_tensor;
    }
    
    inline T* ptr() const { return m_view.ptr(); }
    inline i64 size() const { return m_view.size(); }
    inline i64 size_bytes() const { return m_view.size_bytes(); }
    
    inline const View& view() const { return m_view; }
    inline View& view() { return m_view; }

    inline T& operator[](i64 i) { return ptr()[i]; }
};

} // namespace beta


} // namespace vlm
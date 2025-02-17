#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <ostream>

#include "vlm_types.hpp"

namespace vlm {

// TODO: remove HostDevice
enum class Location {
    Device, Host
};

class Memory {
    public:
        explicit Memory(bool unified) : m_unified(unified) {}
        virtual ~Memory() = default;
        virtual void* alloc(Location location, i64 size) const = 0;
        virtual void free(Location location, void* ptr) const = 0;
        virtual void copy(Location dst_loc, void* dst, i64 dst_stride, Location src_loc, const void* src, i64 src_stride, i64 elem_size, i64 size) const = 0;
        virtual void fill(Location loc, float* ptr, i64 stride, float value, i64 size) const = 0;
        virtual void fill(Location loc, double* ptr, i64 stride, double value, i64 size) const = 0;

        bool is_unified() const { return m_unified; }
    private:
        const bool m_unified;
};

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

template<typename T, i64 Dim, Location L>
class TensorView {
    using DimArray = std::array<i64, Dim>;
    public:
        TensorView() = default;
        explicit TensorView(Memory* memory, const DimArray& shape, const DimArray& stride, T* ptr) : m_memory(memory), m_shape(shape), m_stride(stride), m_ptr(ptr) {}
        ~TensorView() = default;

        inline i64 size() const { return shape_size(m_shape); }
        inline i64 size_bytes() const { return size() * sizeof(T); }
        inline const DimArray& stride() const { return m_stride; }
        inline i64 stride(i64 dim) const { assert(dim < Dim); return m_stride[dim]; }
        inline const DimArray& shape() const { return m_shape; }
        inline i64 shape(i64 dim) const { assert(dim < Dim); return m_shape[dim]; }
        constexpr i64 dim() const { return Dim;}
        constexpr Location location() const { return L; }
        inline T* ptr() const {return m_ptr;}

        template<typename... Idx> constexpr T& operator()(Idx... idx)       { return m_ptr[offset({idx...})]; }
        template<typename... Idx> constexpr T& operator()(Idx... idx) const { return m_ptr[offset({idx...})]; }

        constexpr T& operator()(const DimArray& indices)       { return m_ptr[offset(indices)]; }
        constexpr T& operator()(const DimArray& indices) const { return m_ptr[offset(indices)]; }

        constexpr T& operator[](i64 i)       { return ptr()[i]; }
        constexpr T& operator[](i64 i) const { return ptr()[i]; }

        template<typename... Args>
        constexpr auto slice(Args... args) const {
            constexpr i64 M = CountRanges<Args...>::value;
            static_assert(sizeof...(args) == Dim, "The number of indices must match the dimension N.");
            static_assert(M <= Dim, "Too many ranges provided compared to the view's dimensionality");

            T* newPtr = m_ptr;
            std::array<i64, M> new_shape{};
            std::array<i64, M> newstride{};
            i64 newDimIndex = 0;

            i64 argIndex = 0;
            ([&](auto& arg) {
                if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, Range>) {
                    i64 first = (arg[0] < 0) ? m_shape[argIndex] + arg[0]: arg[0];
                    i64 last = (arg[1] < 0) ? m_shape[argIndex] + arg[1] + 1 : arg[1];
                    assert((first >= 0) && (first < m_shape[argIndex]));
                    assert((last >= 0) && (last <= m_shape[argIndex]));
                    assert(last - first > 0);
                    newPtr += first * m_stride[argIndex];
                    new_shape[newDimIndex] = last - first;
                    newstride[newDimIndex] = m_stride[argIndex];
                    newDimIndex++;
                } else if constexpr (std::is_integral_v<std::decay_t<decltype(arg)>>) {
                    i64 real_arg = (arg < 0) ? m_shape[argIndex] + arg : arg;
                    assert((real_arg >= 0) && (real_arg < m_shape[argIndex]));
                    newPtr += real_arg * m_stride[argIndex];
                }
                argIndex++;
            }(args), ...);
            
            return TensorView<T, M, L>(m_memory, new_shape, newstride, newPtr);
        }

        template<typename... Args>
        constexpr auto reshape(Args... args) const {
            constexpr i32 D = sizeof...(Args);
            std::array<i64, D> new_shape = { static_cast<i64>(args)... };
            std::array<i64, D> new_strides;
            i64 contiguous = shape(0); // fused shape max
            i64 i =  1; // number of initial dims that can be fused
            for (; i < Dim; i++) {
                if (stride(i)/stride(0) != contiguous) break;
                contiguous *= shape(i);
            }

            assert(contiguous % new_shape[0] == 0);
            contiguous /= new_shape[0];
            new_strides[0] = stride(0);

            for (i64 ii = 1; ii < D; ii++) {
                i64 ns = new_shape[ii];

                if (contiguous == 1 && i != Dim) {
                    contiguous = shape(i);
                    ++i;
                }

                assert(contiguous % ns == 0);
                contiguous /= ns;
                
                if (contiguous * ns == shape(i-1)) {
                    new_strides[ii] = stride(i-1);
                } else {
                    new_strides[ii] = new_strides[ii-1] * new_shape[ii-1];
                }
            }

            return TensorView<T, D, L>(m_memory, new_shape, new_strides, m_ptr);
        }

        template<Location ML>
        void to(const TensorView<T, Dim, ML>& dst) const {
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

            std::function<void(i64)> copy_lambda = [&](i64 di) {
                if (di == i-1) {
                    m_memory->copy(
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
                    for (i64 dii = 0; dii < shape(di); dii++) {
                        dim_idx[di] = dii;
                        copy_lambda(di-1);
                    }
                }
            };
            copy_lambda(Dim-1);
        }

        void fill(T value) const {
            i64 contiguous = shape(0);
            i64 i =  1;
            for (; i < Dim; i++) {
                if (stride(i)/stride(0) != contiguous) break; // can put this in for loop body
                contiguous *= shape(i);
            }

            DimArray dim_idx{0};

            std::function<void(i64)> fill_lambda = [&](i64 di) {
                if (di == i-1) {
                    m_memory->fill(
                        this->location(),
                        this->ptr() + this->offset(dim_idx),
                        this->stride(0),
                        value,
                        contiguous
                    );
                } else {
                    for (i64 dii = 0; dii < shape(di); dii++) {
                        dim_idx[di] = dii;
                        fill_lambda(di-1);
                    }
                }
            };
            fill_lambda(Dim-1);
        }
        
        // linear pointer offset to data
        constexpr i64 offset(const DimArray& indices) const {
            i64 index = 0;
            for (i64 i = 0; i < Dim; i++) {
                i64 idx = (indices[i] < 0) ? indices[i] + m_shape[i] : indices[i];
                index += idx * m_stride[i];
            }
            return index;
        }
    
        // linear "virtual" index to data (doesnt take into account strides)
        constexpr i64 to_linear_index(const DimArray& indices) const {
            i64 index = (indices[0] < 0) ? indices[0] + m_shape[0] : indices[0];
            for (i64 i = 1; i < Dim; i++) {
                i64 idx = (indices[i] < 0) ? indices[i] + m_shape[i] : indices[i];
                index += idx * m_shape[i-1];
            }
            return index;
        }

        constexpr i64 shape_size(const DimArray& shape) const {
            i64 size = 1;
            for (i64 i = 0; i < Dim; i++) size *= shape[i];
            return size;
        }

        Memory* m_memory = nullptr;
    private:

        DimArray m_shape;
        DimArray m_stride;
        T* m_ptr = nullptr;
};

namespace detail {
    template<typename T, i64 Dim, Location L>
    void print_tensor_recursive(std::ostream& os, const TensorView<T, Dim, L>& view, 
                              std::array<i64, static_cast<size_t>(Dim)>& indices, i64 current_dim) {
        if (current_dim < 0) {
            os << "(";
            for (i64 i = 0; i < Dim; ++i) {
                os << indices[i];
                if (i < Dim - 1) os << ", ";
            }
            os << ") = " << view(indices) << "\n";
            return;
        }

        for (i64 i = 0; i < view.shape(current_dim); ++i) {
            indices[current_dim] = i;
            print_tensor_recursive(os, view, indices, current_dim - 1);
        }
    }
}

template<typename T, i64 Dim, Location L>
std::ostream& operator<<(std::ostream& os, const TensorView<T, Dim, L>& view) {
    std::array<i64, static_cast<size_t>(Dim)> indices{};
    detail::print_tensor_recursive(os, view, indices, Dim - 1);
    return os;
}
 
template<typename T, int Dim, Location L>
class Tensor {
private:
    using View = TensorView<T, Dim, L>;
    using DimArray = std::array<i64, Dim>;
    View m_view;
    
public:
    explicit Tensor(Memory* memory) : 
        m_view(memory, {}, {}, nullptr) {}
    
    ~Tensor() {
        m_view.m_memory->free(L, m_view.ptr());
    }

    Tensor(const Tensor&) = delete; // no copy constructor
    Tensor& operator=(const Tensor&) = delete; // no copy assignment
    Tensor(Tensor&& other) noexcept 
        : m_view(std::move(other.m_view)) {
        other.m_view = View(m_view.m_memory, {}, {}, nullptr);
    }

    Tensor& operator=(Tensor&& other) noexcept = delete;

    void init(const DimArray& shape) {
        if (shape == m_view.shape()) return; // dont reallocate if same shape
        // TODO: dont reallocate if same size, only change the view shape and strides
        m_view.m_memory->free(L, m_view.ptr());
        i64 size = shape[0];
        DimArray stride;
        stride[0] = 1;
        for (i64 i = 1; i < Dim; ++i) {
            size *= shape[i];
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        auto ptr = static_cast<T*>(m_view.m_memory->alloc(L, size * sizeof(T)));
        m_view = View(m_view.m_memory, shape, stride, ptr);
    }

    [[nodiscard]] Tensor<T, Dim, L> clone() {
        Tensor<T, Dim, L> cloned_tensor(m_view.m_memory);
        cloned_tensor.init(m_view.shape());
        m_view.to(cloned_tensor.view());
        return cloned_tensor; // uses move constructor
    }
    
    inline T* ptr() const { return m_view.ptr(); }
    inline i64 size() const { return m_view.size(); }
    inline i64 size_bytes() const { return m_view.size_bytes(); }
    
    inline const View& view() const { return m_view; }
    inline View& view() { return m_view; }

    inline T& operator[](i64 i) { return ptr()[i]; }
};

template<i32 N> using MultiDim = std::vector<std::array<i64, N>>;

template<typename T, int Dim, Location L>
class MultiTensor {
    public:
        explicit MultiTensor(Memory* memory) : m_memory(memory) {}
        void init(const MultiDim<Dim>& shapes) {
            m_tensors.clear();
            m_tensors.reserve(shapes.size());
            m_tensor_views.resize(shapes.size());
            
            for (i64 i = 0; i < shapes.size(); i++) {
                m_tensors.emplace_back(m_memory);  // Construct Tensor with memory
                m_tensors[i].init(shapes[i]);
                m_tensor_views[i] = m_tensors[i].view();
            }
        }
        std::vector<TensorView<T, Dim, L>>& views() { return m_tensor_views; }
        std::vector<TensorView<T, Dim, L>>& views() const { return m_tensor_views; }

    private:
        Memory* m_memory;
        std::vector<Tensor<T, Dim, L>> m_tensors;
        std::vector<TensorView<T, Dim, L>> m_tensor_views;
};

template<Location L> using Tensor1D = Tensor<f32, 1, L>;
template<Location L> using Tensor2D = Tensor<f32, 2, L>;
template<Location L> using Tensor3D = Tensor<f32, 3, L>;
template<Location L> using TensorView1D = TensorView<f32, 1, L>;
template<Location L> using TensorView2D = TensorView<f32, 2, L>;
template<Location L> using TensorView3D = TensorView<f32, 3, L>;
template<Location L> using MultiTensor1D = MultiTensor<f32, 1, L>;
template<Location L> using MultiTensor2D = MultiTensor<f32, 2, L>;
template<Location L> using MultiTensor3D = MultiTensor<f32, 3, L>;
template<typename T, int Dim, Location L> using MultiTensorView = std::vector<TensorView<T, Dim, L>>;
template<Location L> using MultiTensorView1D = MultiTensorView<f32, 1, L>;
template<Location L> using MultiTensorView2D = MultiTensorView<f32, 2, L>;
template<Location L> using MultiTensorView3D = MultiTensorView<f32, 3, L>;

} // namespace vlm
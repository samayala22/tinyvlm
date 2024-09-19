#include <array>
#include <type_traits>
#include <cassert>

#ifdef _MSC_VER
#define TINY_NO_INLINE __declspec(noinline)
#define TINY_INLINE _forceinline
#else
#define TINY_NO_INLINE __attribute__((noinline))
#define TINY_INLINE __attribute__((always_inline))
#endif

namespace tiny {

using Range = std::array<int, 2>;

constexpr Range all{0, -1};

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

template<typename T, std::size_t N> 
class View {
public:
    constexpr View(T* const ptr, const std::array<std::size_t, N>& dims) 
        : _ptr(ptr), _dims(dims), _strides(compute_strides(dims)) {}

    constexpr View(T* const ptr, const std::array<std::size_t, N>& dims, const std::array<std::size_t, N>& strides) 
        : _ptr(ptr), _dims(dims), _strides(strides) {}

    constexpr View(const View& other) 
        : _ptr(other._ptr), _dims(other._dims), _strides(other._strides) {}

    template<typename... Idx>
    TINY_INLINE T& operator()(std::size_t first, Idx... idx) const { return _ptr[compute_index(first, idx...)]; }

    template<typename... Idx>
    TINY_INLINE T* ptr(std::size_t first, Idx... idx) const { return _ptr + compute_index(first, idx...); }

    template<typename... Args>
    TINY_INLINE auto view(Args... args) { // Removed 'constexpr'
        constexpr std::size_t M = CountRanges<Args...>::value;
        static_assert(sizeof...(args) == N, "The number of indices must match the dimension N.");
        static_assert(M <= N, "Too many ranges provided compared to the view's dimensionality");

        T* newPtr = _ptr;
        std::array<std::size_t, M> newDims{};
        std::array<std::size_t, M> newStrides{};
        std::size_t newDimIndex = 0;

        std::size_t argIndex = 0;
        ([&](auto& arg) {
            if constexpr (std::is_same<std::decay_t<decltype(arg)>, Range>::value) {
                std::size_t first = arg[0]; // Removed 'constexpr'
                std::size_t last = (arg[1] < 0) ? _dims[argIndex] + arg[1] + 1 : arg[1];
                assert((first >= 0) && (first < _dims[argIndex]));
                assert((last >= 0) && (last <= _dims[argIndex])); // Changed '<' to '<=' for upper bound
                assert(last - first > 0);
                newPtr += first * _strides[argIndex];
                newDims[newDimIndex] = last - first;
                newStrides[newDimIndex] = _strides[argIndex];
                newDimIndex++;
            } else if constexpr (std::is_integral<std::decay_t<decltype(arg)>>::value) {
                std::size_t real_arg = (arg < 0) ? _dims[argIndex] + arg : arg; // Removed '+1' as indexing starts from 0
                assert((real_arg >= 0) && (real_arg < _dims[argIndex]));
                newPtr += real_arg * _strides[argIndex];
            }
            argIndex++;
        }(args), ...);

        return View<T, M>(newPtr, newDims, newStrides);
    }

    TINY_INLINE T* data() const { return _ptr; }
    TINY_INLINE View& operator=(const View& other) = default;

    TINY_INLINE const std::size_t& dims(std::size_t idx) const { return _dims[idx]; } 
    TINY_INLINE const std::size_t& strides(std::size_t idx) const { return _strides[idx]; } 

private:
    template<typename... Idx>
    std::size_t compute_index(std::size_t first, Idx... idx) const { // Removed 'constexpr'
        static_assert(sizeof...(idx) == N-1, "The number of indices must match the dimension N.");
        static_assert((std::is_integral<std::decay_t<Idx>>::value && ...), "All indices must be integers");

        #ifndef NDEBUG
        std::size_t ii = 0; // Changed from 1 to 0 to match zero-based indexing
        ((assert(idx >= 0 && idx < _dims[ii++])), ...);
        #endif

        std::size_t index = first;
        std::size_t i = 1;
        ((index += idx * _strides[i++]), ...);
        return index;
    }

    static std::array<std::size_t, N> compute_strides(const std::array<std::size_t, N>& dims) {
        std::array<std::size_t, N> s{};
        s[0] = 1;
        for (std::size_t i = 1; i < N; ++i) {
            s[i] = s[i - 1] * dims[i - 1];
        }
        return s;
    }

    T* const _ptr;
    const std::array<std::size_t, N> _dims;
    const std::array<std::size_t, N> _strides;
};

} // namespace tiny

// #include <cstdio>
// using namespace tiny;

// int main() {
//     constexpr int n = 3;
//     float a[n * n * n] = {};

//     for (int i = 0; i < n * n * n; i++) {
//         a[i] = static_cast<float>(i);
//     }

//     View<float, 3> av{&a[0], {static_cast<std::size_t>(n), static_cast<std::size_t>(n), static_cast<std::size_t>(n)}};
    
//     View<float, 2> bv = av.view(all, -2, Range{0, 3});

//     for (std::size_t j = 0; j < bv.dims(1); j++) {
//         for (std::size_t i = 0; i < bv.dims(0); i++) {
//             std::printf("%f ", bv(i, j));
//         }
//         std::printf("\n");
//     }
    
//     assert(bv.dims(0) == 3);
//     assert(bv.dims(1) == 3);
//     assert(bv(0,0) == 3.f);
//     assert(bv(2,1) == 14.f);
//     assert(bv(2,2) == 23.f);

//     return 0;
// }
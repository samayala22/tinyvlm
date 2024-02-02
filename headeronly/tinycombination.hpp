#pragma once

// MIT license
// Adapted from https://stackoverflow.com/a/53217310

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <array>
#include <functional> // For std::reference_wrapper

namespace tiny {

// Odometer style combination counter
template <std::size_t N>
bool increase(const std::array<std::size_t, N>& sizes, std::array<std::size_t, N>& it) {
    for (std::size_t i = 0; i != N; ++i) {
        const std::size_t index = N - 1 - i;
        ++it[index];
        if (it[index] == sizes[index]) {
            it[index] = 0;
        } else {
            return true;
        }
    }
    return false;
}

// Applies a given function to the elements of the vectors at the current indices.
// The function f is applied to the elements pointed to by the indices in 'it'.
template <typename F, std::size_t ... Is, std::size_t N, typename Tuple>
void apply_impl(F&& f,
                std::index_sequence<Is...>,
                const std::array<std::size_t, N>& it,
                const Tuple& tuple) {
    // Calls the function 'f' with the elements of 'tuple' at indices specified in 'it'.
    // std::get<Is> gets the vector at index Is in the tuple, and [it[Is]] accesses the element at the index in that vector.
    f(std::ref(std::get<Is>(tuple)[it[Is]])...);
}

// Iterates over all combinations of elements from the input vectors and applies a function to each combination.
template <typename F, typename ... Ts>
void combination_apply(F&& f, const std::vector<Ts>&... vs) {
    constexpr std::size_t N = sizeof...(Ts); // Number of input vectors.
    std::array<std::size_t, N> sizes{{vs.size()...}}; // Array holding the sizes of each vector.
    std::array<std::size_t, N> it{}; // Array to keep track of current indices in each vector.

    // Iterate over all combinations. The 'increase' function updates 'it' to the next combination of indices.
    do {
        // Apply the function 'f' to the current combination of elements.
        apply_impl(f, std::index_sequence_for<Ts...>(), it, std::tie(vs...));
    } while (increase(sizes, it)); // Continue until all combinations have been processed.
}

/// @brief Creates a combination of all input vectors into a vector of tuples of references to the original vector members
/// @param vs Arbitrary amount of vectors of any size and type
/// @return Returns a vector of tuples containing all combinations of the input vectors
template <typename ... Ts>
std::vector<std::tuple<std::reference_wrapper<const Ts>...>> make_combination(const std::vector<Ts>&... vs) {
    std::vector<std::tuple<std::reference_wrapper<const Ts>...>> res;

    combination_apply([&res](const auto&... args) { res.emplace_back(args...); }, vs...);
    return res;
}

} // namespace tiny

// Example usage:
// int main() {
//     std::vector<int> v1 = {1, 2, 3, 4};
//     std::vector<std::string> v2 = {"a", "b"};
//     std::vector<float> v3 = {0.1f, 0.2f};

//     auto combinations = tiny::make_combination(v1, v2, v3);
//     for (const auto& [v_int, v_string, v_float]: combinations) {
//         std::cout << "int: " << v_int << "| string: " << v_string.get() << "| float: " << v_float << "\n";
//     }

//     return 0;
// }


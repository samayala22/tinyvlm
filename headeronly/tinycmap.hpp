#pragma once

#include <array>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace tiny {

template <typename Key, typename Value, std::size_t Size> struct Map {
    std::array<std::pair<Key, Value>, Size> data;

    template <typename It, std::size_t... I>
    constexpr Map(It begin, std::index_sequence<I...>)
        : data{*(begin + I)...} {}

    [[nodiscard]] constexpr Value at(const Key key) const {
        for (const auto &elem : data) {
            if (elem.first == key)
                return elem.second;
        }
        throw std::range_error("Key not Found");
    }

    constexpr Value operator[](const Key key) const { return at(key); }

    constexpr std::size_t size() const { return Size; }
};

template <typename Key, typename Value, std::size_t Size>
constexpr auto make_map(std::pair<Key, Value> (&&m)[Size]) {
    return Map<Key, Value, Size>(std::begin(m),
                                 std::make_index_sequence<Size>{});
}

}; // namespace tiny

using namespace std::literals::string_view_literals;

// int main(int argc, char** argv) {
//   static constexpr auto color_values = tiny::make_map<std::string_view,
//   int>({
//   {"black"sv, 7}, {"blue"sv, 3},  {"cyan"sv, 5},   {"green"sv, 2},
//   {"magenta"sv, 6}, {"red"sv, 1}, {"white"sv, 8}, {"yellow"sv, 4}});

//   std::cout << color_values["red"] << std::endl;
//   std::cout << color_values["magenta"] << std::endl;
//   std::cout << color_values["yellow"] << std::endl;
//   std::cout << color_values.size() << std::endl;
//   return 0;
// }

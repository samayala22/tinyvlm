#pragma once

#include <chrono>
#include <cstdio>

namespace tiny {

constexpr static double us_in_s = 1'000'000;
constexpr static double us_in_ms = 1'000;

class ScopedTimer {
public:
    ScopedTimer(const char* name) : m_name(name), m_start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        const auto m_end = std::chrono::high_resolution_clock::now();
        const auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start).count());
        std::printf("%s: ", m_name);

        if (duration > us_in_s) {
            std::printf("%.2f s\n", duration / us_in_s);
        } else if (duration > us_in_ms) {
            std::printf("%.0f ms\n", duration / us_in_ms);
        } else {
            std::printf("%.0f us\n", duration);
        }
    }
private:
    const char* m_name; // string literal
    const std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

}
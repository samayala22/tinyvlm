#pragma once

#include <chrono>
#include <iostream>

class SimpleTimer {
public:
    SimpleTimer(const char* name) : m_Name(name) {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();
    }

    ~SimpleTimer() { Stop();}

    void Stop() {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
        long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

        auto duration = end - start;
        std::cout << m_Name << ": ";

        if (duration > 1e6) {
            std::cout << duration * 1e-6 << " s" << std::endl;
        } else if (duration > 1e3) {
            std::cout << duration * 1e-3 << " ms" << std::endl;
        } else {
            std::cout << duration << " us" << std::endl;
        }
    }
private:
    const char* m_Name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
};

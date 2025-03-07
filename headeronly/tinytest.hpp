#pragma once 

#include <iostream>

#define TINY_ASSERT_EQ(x, y) \
    do { \
        auto val1 = (x); \
        auto val2 = (y); \
        if (!(val1 == val2)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": " << #x << " == " << #y << " (Left: " << val1 << ", Right: " << val2 << ")\n"; \
            std::abort(); \
        } \
    } while (0)

#define TINY_ASSERT_NEAR(x, y, tol) \
    do { \
        auto val1 = (x); \
        auto val2 = (y); \
        if (!(std::abs(val1 - val2) <= tol)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": |" << #x << " - " << #y << "| <= " << tol << " (Left: " << val1 << ", Right: " << val2 << ", Diff: " << std::abs(val1 - val2) << ")\n"; \
            std::abort(); \
        } \
    } while (0)

#define TINY_ABORT(msg) \
    do { \
        std::cerr << "Aborting at " << __FILE__ << ":" << __LINE__ << ": " << msg << "\n"; \
        std::abort(); \
    } while (0)
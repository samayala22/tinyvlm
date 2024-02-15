// This header should not be included by another header file, only by source files.
#pragma once

#include <taskflow/taskflow.hpp> // includes <thread>, <mutex>, <memory>

// Taskflow executor singleton wrapper (not thread-safe by design)
namespace vlm {
    class Executor {
    public:
        Executor(const Executor&) = delete;
        Executor& operator=(const Executor&) = delete;
        static tf::Executor& instance(size_t num_threads) {
            if (!_instance) _instance = std::make_unique<tf::Executor>(num_threads ? num_threads : std::thread::hardware_concurrency());
            return *_instance;
        }
        static tf::Executor& get() {
            if (!_instance) return instance(0);
            return *_instance;
        }
    private:
        inline static std::unique_ptr<tf::Executor> _instance;

        Executor() = default;
        ~Executor() = default;
    };
}
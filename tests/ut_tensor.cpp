#include "vlm_backend.hpp"
#include "vlm_memory.hpp"

#include <cstdio>
#include <cassert>

#define CHECK(condition)                     \
    do {                                            \
        if (!(condition)) {                         \
            std::fprintf(stderr,                         \
                    "Assertion failed: %s\n"        \
                    "File: %s, Line: %d\n",        \
                    #condition, __FILE__, __LINE__);\
            std::abort();                                \
        }                                           \
    } while (0)

using namespace vlm;

int main(int, char**) {
    const std::vector<std::string> backends = get_available_backends();

    for (const auto& backend_name : backends) {
        const std::unique_ptr<Backend> backend = create_backend(backend_name);
        const std::unique_ptr<Memory> memory = backend->create_memory_manager();

        Tensor<f32, 3, Location::Host> tensor_h{memory.get()};
        Tensor<f32, 3, Location::Device> tensor_d{memory.get()};

        const i64 m = 3;
        const i64 n = 3;
        const i64 k = 4;
        tensor_d.init({m, n, k});
        tensor_h.init({m, n, k});

        auto& tdv = tensor_d.view();
        auto& thv = tensor_h.view();
        assert(tdv.shape() == thv.shape());

        for (u32 i = 0; i < tensor_h.size(); i++) {
            tensor_h[i] = static_cast<float>(i);
        }

        thv.to(tdv);

        {
            auto bv = thv.slice(All, 1, Range{0, 3});

            CHECK(bv(0, 0) == 3.0f);
            CHECK(bv(1, 0) == 4.0f);
            CHECK(bv(2, 1) == 14.0f);
            CHECK(bv(2, 2) == 23.0f);
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(All, Range{0, 2}, 0);
            auto b = tv.slice(All, Range{1, 3}, 2);
            a.to(b);
            tv.to(thv);

            CHECK(thv(0, 2, 2) == 3.0f);
            CHECK(thv(1, 1, 2) == 1.0f);
            CHECK(thv(2, 2, 2) == 5.0f);
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(0, All, All);
            auto b = tv.slice(1, All, All);
            a.to(b);
            tv.to(thv);

            CHECK(thv(0, 0, 0) == thv(1, 0, 0));
            CHECK(thv(0, 1, 1) == thv(1, 1, 1));
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(0, All, All);
            auto b = tv.slice(All, 2, All);

            auto aa = a.slice(Range{0,2}, All);
            auto bb = b.slice(Range{1,3}, All);
            aa.to(bb);
            tv.to(thv);

            CHECK(thv(0,0,1) == thv(1, 2, 1));
            CHECK(thv(0,0,2) == thv(1, 2, 2));
        }

        { // orthogonal slices
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(0, All, Range{0,2});
            auto b = tv.slice(All, Range{1,3}, 2);
            
            a.to(b);
            tv.to(thv);

            CHECK(thv(0,0,0) == thv(0, 1, 2));
            CHECK(thv(0,1,0) == thv(1, 1, 2));
            CHECK(thv(0,2,1) == thv(2, 2, 2));
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(Range{0,2}, Range{1,3}, All);
            auto b = a.reshape(1, 2, 2, 2, 2);
            
            const std::array<i64, 5> correct_stride{1, 1, 3, 9, 18};
            CHECK(b.stride() == correct_stride);
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(All, Range{1,3}, All);
            auto b = a.reshape(2, 3, 1, 1, 4);

            const std::array<i64, 5> correct_stride{1, 2, 9, 9, 9};
            CHECK(b.stride() == correct_stride);
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(0, All, All);
            auto b = a.reshape(3, 2, 2);

            const std::array<i64, 3> correct_stride{3, 9, 18};
            CHECK(b.stride() == correct_stride);
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(All, All, All);
            auto b = a.reshape(2, 18);

            const std::array<i64, 2> correct_stride{1, 2};
            CHECK(b.stride() == correct_stride);
        }

        {
            auto t = tensor_d.clone();
            const auto& tv = t.view();
            auto a = tv.slice(0, All, All);
            a.fill(111.f);
            tv.to(thv);

            CHECK(thv(0,0,0) == 111.f);
            CHECK(thv(0,1,1) == 111.f);
            CHECK(thv(0,2,2) == 111.f);
        }

        {
            Tensor<f32, 1, Location::Host> t{memory.get()};
            t.init({10});
            const auto& tv = t.view();
            auto b = tv.reshape(10, 1, 1);

            const std::array<i64, 3> correct_stride{1, 10, 10};
            CHECK(b.stride() == correct_stride);
        }
    }

    return 0;
}
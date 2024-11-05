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

void print3d(const TensorView<f32, 3, Location::Host>& tensor) {
    // Print the 3D tensor
    for (i64 z = 0; z < tensor.shape(2); z++) {
        std::printf("Layer %lld:\n", z);
        for (i64 x = 0; x < tensor.shape(0); x++) {
            for (i64 y = 0; y < tensor.shape(1); y++) {
                std::printf("%6.1f ", tensor(x, y, z));
            }
            std::printf("\n");
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    const std::vector<std::string> backends = get_available_backends();

    for (const auto& backend_name : backends) {
        std::unique_ptr<Backend> backend = create_backend(backend_name);
        std::unique_ptr<Memory> memory = backend->create_memory_manager();

        Tensor<f32, 3, Location::Host> tensor_h{*memory};
        Tensor<f32, 3, Location::Device> tensor_d{*memory};

        const i64 n = 3;
        tensor_d.init({n, n, n});
        tensor_h.init({n, n, n});

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
            auto tv = t.view();
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
            auto tv = t.view();
            auto a = tv.slice(0, All, All);
            auto b = tv.slice(1, All, All);
            a.to(b);
            tv.to(thv);

            CHECK(thv(0, 0, 0) == thv(1, 0, 0));
            CHECK(thv(0, 1, 1) == thv(1, 1, 1));
        }
    }

    return 0;
}
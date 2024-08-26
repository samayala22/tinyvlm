#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "tinycpuid.hpp"

#include <string>

#ifdef VLM_CPU
#include "vlm_backend_cpu.hpp"
#endif
#ifdef VLM_CUDA
#include "vlm_backend_cuda.hpp"
#endif

using namespace vlm;

std::unique_ptr<Backend> vlm::create_backend(const std::string& backend_name) {
    //tiny::CPUID cpuid;
    //cpuid.print_info();

    #ifdef VLM_CPU
    if (backend_name == "cpu") {
        return std::make_unique<BackendCPU>();
    }
    #endif
    #ifdef VLM_CUDA
    if (backend_name == "cuda") {
        return std::make_unique<BackendCUDA>();
    }
    #endif
    throw std::runtime_error("Unsupported backend: " + backend_name); // TODO: remove
}

std::vector<std::string> vlm::get_available_backends() {
    std::vector<std::string> backends;
    #ifdef VLM_CPU
    backends.push_back("cpu");
    #endif
    #ifdef VLM_CUDA
    backends.push_back("cuda");
    #endif
    return backends;
}

Backend::~Backend() {
    // Free device-device ptrs
    memory->free(MemoryLocation::Device, d_solver_info);
    memory->free(MemoryLocation::Device, d_solver_ipiv);
    memory->free(MemoryLocation::Device, d_solver_buffer);
    memory->free(MemoryLocation::Device, d_val);
}
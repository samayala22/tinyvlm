#include "vlm_backend.hpp"
#include "vlm_mesh.hpp"
#include "tinycpuid.hpp"

#include <string>

#ifdef VLM_AVX2
#include "vlm_backend_avx2.hpp"
#endif
#ifdef VLM_CUDA
#include "vlm_backend_cuda.hpp"
#endif

using namespace vlm;

f32 Backend::compute_coefficient_cl(const FlowData& flow) {
    return compute_coefficient_cl(flow, mesh.s_ref, 0, mesh.ns);
}

linalg::alias::float3 Backend::compute_coefficient_cm(const FlowData& flow) {
    return compute_coefficient_cm(flow, mesh.s_ref, mesh.c_ref, 0, mesh.ns);
}

f32 Backend::compute_coefficient_cd(const FlowData& flow) {
    return compute_coefficient_cd(flow, mesh.s_ref, 0, mesh.ns);
}

std::unique_ptr<Backend> vlm::create_backend(const std::string& backend_name, Mesh& mesh) {
    tiny::CPUID cpuid;
    cpuid.print_info();

    #ifdef VLM_AVX2
    if (backend_name == "avx2" && cpuid.has("AVX2")) {
        return std::make_unique<BackendAVX2>(mesh);
    }
    #endif
    #ifdef VLM_CUDA
    if (backend_name == "cuda") {
        return std::make_unique<BackendCUDA>(mesh);
    }
    #endif
    throw std::runtime_error("Unsupported backend: " + backend_name);
}
#include "vlm_backend_cuda.hpp"
#include <cusolverDn.h>

using namespace vlm;

#define CHECK_CUSOLVER(call) \
    do { \
        const cusolverStatus_t err = (call); \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSolver Error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class CUSolverCtx {
private:
    CUSolverCtx() {
        CHECK_CUSOLVER(cusolverDnCreate(&m_dn_handle));
        CHECK_CUSOLVER(cusolverDnSetStream(m_dn_handle, 0)); // mmmmh
    };
    ~CUSolverCtx() {
        CHECK_CUSOLVER(cusolverDnDestroy(m_dn_handle));
    }
    cusolverDnHandle_t m_dn_handle;

public:
    CUSolverCtx(const CUSolverCtx&) = delete;
    CUSolverCtx& operator=(const CUSolverCtx&) = delete;

    static CUSolverCtx& get() {
        static CUSolverCtx instance;
        return instance;
    }

    cusolverDnHandle_t& dn_handle() { return m_dn_handle; }
};

class CUDA_LU final : public LU {
    public:
        CUDA_LU(std::unique_ptr<Memory> memory);
        ~CUDA_LU() = default;

        void init(const TensorView<f32, 2, Location::Device>& A) override;
        void factorize(const TensorView<f32, 2, Location::Device>& A) override;
        void solve(const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& x) override;
    private:
        Tensor<i32, 1, Location::Device> ipiv{m_memory.get()};
        Tensor<f32, 1, Location::Device> buffer{m_memory.get()};
        Tensor<i32, 1, Location::Device> info_d{m_memory.get()}; // single value
        Tensor<i32, 1, Location::Host> info_h{m_memory.get()}; // single value
};

std::unique_ptr<LU> BackendCUDA::create_lu_solver() { return std::make_unique<CUDA_LU>(create_memory_manager()); }

CUDA_LU::CUDA_LU(std::unique_ptr<Memory> memory) : LU(std::move(memory)) {}

void CUDA_LU::init(const TensorView<f32, 2, Location::Device>& A) {
    int bufsize = 0;
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(CUSolverCtx::get().dn_handle(), A.shape(0), A.shape(1), A.ptr(), A.stride(1), &bufsize));
    info_d.init({1});
    info_h.init({1});
    buffer.init({bufsize});
    ipiv.init({std::min(A.shape(0), A.shape(1))});
}

void CUDA_LU::factorize(const TensorView<f32, 2, Location::Device>& A) {
    CHECK_CUSOLVER(cusolverDnSgetrf(
        CUSolverCtx::get().dn_handle(),
        A.shape(0),
        A.shape(1),
        A.ptr(),
        A.stride(1),
        buffer.ptr(),
        ipiv.ptr(),
        info_d.ptr()
    ));
    info_d.view().to(info_h.view());

    if (info_h[0] != 0) std::printf("Error: LU factorization failed\n"); // todo: stderr ?
}

void CUDA_LU::solve(const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& x) {
    CHECK_CUSOLVER(cusolverDnSgetrs(
        CUSolverCtx::get().dn_handle(),
        CUBLAS_OP_N,
        A.shape(1),
        x.shape(1),
        A.ptr(),
        A.stride(1),
        ipiv.ptr(),
        x.ptr(),
        x.stride(1),
        info_d.ptr()
    ));
    info_d.view().to(info_h.view());
    if (info_h[0] != 0) std::printf("Error: LU solve failed\n"); // todo: stderr ?
}
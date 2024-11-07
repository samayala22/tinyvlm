#include "vlm_backend_cuda.hpp"
#include <cublas_v2.h>

using namespace vlm;

#define CHECK_CUBLAS(call) \
    do { \
        const cublasStatus_t err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class CUBlasCtx {
private:
    CUBlasCtx() {
        CHECK_CUBLAS(cublasCreate(&m_handle));
        CHECK_CUBLAS(cublasSetStream(m_handle, 0));
    };
    ~CUBlasCtx() {
        CHECK_CUBLAS(cublasDestroy(m_handle));
    }
    cublasHandle_t m_handle;

public:
    CUBlasCtx(const CUBlasCtx&) = delete;
    CUBlasCtx& operator=(const CUBlasCtx&) = delete;

    static CUBlasCtx& get() {
        static CUBlasCtx instance;
        return instance;
    }

    cublasHandle_t& handle() { return m_handle; }
};

class CUDA_BLAS final : public BLAS {
    public:
        CUDA_BLAS() = default;
        ~CUDA_BLAS() = default;

        void gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, TensorView<f32, 1, Location::Device>& y, Trans trans = Trans::No) override;
        void gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, TensorView<f32, 2, Location::Device>& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) override;
        void axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, TensorView<f32, 1, Location::Device>& y) override;
        void axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, TensorView<f32, 2, Location::Device>& y) override;

};

std::unique_ptr<BLAS> BackendCUDA::create_blas() { return std::make_unique<CUDA_BLAS>(); }

cublasOperation_t cublas_trans(BLAS::Trans trans) {
    switch (trans) {
        case BLAS::Trans::No: return CUBLAS_OP_N;
        case BLAS::Trans::Yes: return CUBLAS_OP_T;
    }
}

void CUDA_BLAS::gemv(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 1, Location::Device>& x, const f32 beta, TensorView<f32, 1, Location::Device>& y, Trans trans) {
    // TODO: double check if this is correct
    i64 m = (trans == Trans::No) ? A.shape(0) : A.shape(1);
    i64 n = (trans == Trans::No) ? A.shape(1) : A.shape(0);

    CHECK_CUBLAS(cublasSgemv_64(
        CUBlasCtx::get().handle(),
        cublas_trans(trans),
        m,
        n,
        &alpha,
        A.ptr(),
        A.stride(1),
        x.ptr(),
        x.stride(0),
        &beta,
        y.ptr(),
        y.stride(0)
    ));
}

void CUDA_BLAS::gemm(const f32 alpha, const TensorView<f32, 2, Location::Device>& A, const TensorView<f32, 2, Location::Device>& B, const f32 beta, TensorView<f32, 2, Location::Device>& C, Trans trans_a, Trans trans_b) {
    i64 m = (trans_a == BLAS::Trans::No) ? A.shape(0) : A.shape(1);
    i64 n = (trans_b == BLAS::Trans::No) ? B.shape(1) : B.shape(0);
    i64 k = (trans_a == BLAS::Trans::No) ? A.shape(1) : A.shape(0);

    CHECK_CUBLAS(cublasSgemm_64(
        CUBlasCtx::get().handle(),
        cublas_trans(trans_a),
        cublas_trans(trans_b),
        m,
        n,
        k,
        &alpha,
        A.ptr(),
        A.stride(1),
        B.ptr(),
        B.stride(1),
        &beta,
        C.ptr(),
        C.stride(1)
    ));
}

void CUDA_BLAS::axpy(const f32 alpha, const TensorView<f32, 1, Location::Device>& x, TensorView<f32, 1, Location::Device>& y) {
    CHECK_CUBLAS(cublasSaxpy_64(
        CUBlasCtx::get().handle(),
        x.size(),
        &alpha,
        x.ptr(),
        x.stride(0),
        y.ptr(),
        y.stride(0)
    ));
}

void CUDA_BLAS::axpy(const f32 alpha, const TensorView<f32, 2, Location::Device>& x, TensorView<f32, 2, Location::Device>& y) {
    f32 beta = 1.0f;
    CHECK_CUBLAS(cublasSgeam_64(
        CUBlasCtx::get().handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        x.shape(0),
        x.shape(1),
        &alpha,
        x.ptr(),
        x.stride(1),
        &beta,
        y.ptr(),
        y.stride(1),
        y.ptr(),
        y.stride(1)
    ));
}

#include "vlm_backend_cpu.hpp"
#include <cblas.h>

using namespace vlm;

CBLAS_TRANSPOSE trans_to_cblas(BLAS::Trans trans) {
    switch (trans) {
        case BLAS::Trans::No: return CblasNoTrans;
        case BLAS::Trans::Yes: return CblasTrans;
    }
}

class CPU_BLAS final : public BLAS {
    public:
        explicit CPU_BLAS() = default;
        ~CPU_BLAS() override= default;

        void gemv(const f32 alpha, const TensorView2fD& A, const TensorView1fD& x, const f32 beta, const TensorView1fD& y, Trans trans = Trans::No) override;
        void gemm(const f32 alpha, const TensorView2fD& A, const TensorView2fD& B, const f32 beta, const TensorView2fD& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) override;
        void axpy(const f32 alpha, const TensorView1fD& x, const TensorView1fD& y) override;
        void axpy(const f32 alpha, const TensorView2fD& x, const TensorView2fD& y) override;
        void scal(const f32 alpha, const TensorView1fD& x) override;
        f32 norm(const TensorView1fD& x) override;

        void gemv(const f64 alpha, const TensorView2dD& A, const TensorView1dD& x, const f64 beta, const TensorView1dD& y, Trans trans = Trans::No) override;
        void gemm(const f64 alpha, const TensorView2dD& A, const TensorView2dD& B, const f64 beta, const TensorView2dD& C, Trans trans_a = Trans::No, Trans trans_b = Trans::No) override;
        void axpy(const f64 alpha, const TensorView1dD& x, const TensorView1dD& y) override;
        void axpy(const f64 alpha, const TensorView2dD& x, const TensorView2dD& y) override; // Y = alpha *  overrideY
        void scal(const f64 alpha, const TensorView1dD& x) override;
        f64 norm(const TensorView1dD& x) override;
};

f32 CPU_BLAS::norm(const TensorView1fD& x) {
    return cblas_snrm2(x.shape(0), x.ptr(), x.stride(0));
}

void CPU_BLAS::scal(const f32 alpha, const TensorView1fD& x) {
    cblas_sscal(x.shape(0), alpha, x.ptr(), x.stride(0));
}

void CPU_BLAS::gemv(const f32 alpha, const TensorView2fD& A, const TensorView1fD& x, const f32 beta, const TensorView1fD& y, Trans trans) {
    assert(A.stride(0) == 1);

    i32 m = (trans == BLAS::Trans::No) ? A.shape(0) : A.shape(1);
    i32 n = (trans == BLAS::Trans::No) ? A.shape(1) : A.shape(0);

    cblas_sgemv(
        CblasColMajor,
        trans_to_cblas(trans),
        m,
        n,
        alpha,
        A.ptr(),
        static_cast<int>(A.stride(1)),
        x.ptr(),
        static_cast<int>(x.stride(0)),
        beta,
        y.ptr(),
        static_cast<int>(y.stride(0))
    );
}

void CPU_BLAS::gemm(const f32 alpha, const TensorView2fD& A, const TensorView2fD& B, const f32 beta, const TensorView2fD& C, Trans trans_a, Trans trans_b) {
    assert(A.stride(0) == 1);
    assert(B.stride(0) == 1);
    assert(C.stride(0) == 1);

    i32 m = (trans_a == BLAS::Trans::No) ? A.shape(0) : A.shape(1);
    i32 n = (trans_b == BLAS::Trans::No) ? B.shape(1) : B.shape(0);
    i32 k = (trans_a == BLAS::Trans::No) ? A.shape(1) : A.shape(0);

    cblas_sgemm(
        CblasColMajor,
        trans_to_cblas(trans_a),
        trans_to_cblas(trans_b),
        m,
        n,
        k,
        alpha,
        A.ptr(),
        static_cast<int>(A.stride(1)),
        B.ptr(),
        static_cast<int>(B.stride(1)),
        beta,
        C.ptr(),
        static_cast<int>(C.stride(1))
    );
}

void CPU_BLAS::axpy(const f32 alpha, const TensorView1fD& x, const TensorView1fD& y) {
    cblas_saxpy(
        x.shape(0),
        alpha, 
        x.ptr(),
        x.stride(0),
        y.ptr(),
        y.stride(0)
    );
}

void CPU_BLAS::axpy(const f32 alpha, const TensorView2fD& x, const TensorView2fD& y) {
    assert(x.shape() == y.shape());
    for (i64 j = 0; j < x.shape(1); j++) {
        axpy(alpha, x.slice(All, j), y.slice(All, j));
    }
}

f64 CPU_BLAS::norm(const TensorView1dD& x) {
    return cblas_dnrm2(x.shape(0), x.ptr(), x.stride(0));
}

void CPU_BLAS::scal(const f64 alpha, const TensorView1dD& x) {
    cblas_dscal(x.shape(0), alpha, x.ptr(), x.stride(0));
}

void CPU_BLAS::gemv(const f64 alpha, const TensorView2dD& A, const TensorView1dD& x, const f64 beta, const TensorView1dD& y, Trans trans) {
    assert(A.stride(0) == 1);

    i32 m = (trans == BLAS::Trans::No) ? A.shape(0) : A.shape(1);
    i32 n = (trans == BLAS::Trans::No) ? A.shape(1) : A.shape(0);

    cblas_dgemv(
        CblasColMajor,
        trans_to_cblas(trans),
        m,
        n,
        alpha,
        A.ptr(),
        static_cast<int>(A.stride(1)),
        x.ptr(),
        static_cast<int>(x.stride(0)),
        beta,
        y.ptr(),
        static_cast<int>(y.stride(0))
    );
}

void CPU_BLAS::gemm(const f64 alpha, const TensorView2dD& A, const TensorView2dD& B, const f64 beta, const TensorView2dD& C, Trans trans_a, Trans trans_b) {
    assert(A.stride(0) == 1);
    assert(B.stride(0) == 1);
    assert(C.stride(0) == 1);

    i32 m = (trans_a == BLAS::Trans::No) ? A.shape(0) : A.shape(1);
    i32 n = (trans_b == BLAS::Trans::No) ? B.shape(1) : B.shape(0);
    i32 k = (trans_a == BLAS::Trans::No) ? A.shape(1) : A.shape(0);

    cblas_dgemm(
        CblasColMajor,
        trans_to_cblas(trans_a),
        trans_to_cblas(trans_b),
        m,
        n,
        k,
        alpha,
        A.ptr(),
        static_cast<int>(A.stride(1)),
        B.ptr(),
        static_cast<int>(B.stride(1)),
        beta,
        C.ptr(),
        static_cast<int>(C.stride(1))
    );
}

void CPU_BLAS::axpy(const f64 alpha, const TensorView1dD& x, const TensorView1dD& y) {
    cblas_daxpy(
        x.shape(0),
        alpha, 
        x.ptr(),
        x.stride(0),
        y.ptr(),
        y.stride(0)
    );
}

void CPU_BLAS::axpy(const f64 alpha, const TensorView2dD& x, const TensorView2dD& y) {
    assert(x.shape() == y.shape());
    for (i64 j = 0; j < x.shape(1); j++) {
        axpy(alpha, x.slice(All, j), y.slice(All, j));
    }
}

std::unique_ptr<BLAS> BackendCPU::create_blas() { return std::make_unique<CPU_BLAS>(); }

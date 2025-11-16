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

        void init(const TensorView2fD& A) override;
        void factorize(const TensorView2fD& A) override;
        void solve(const TensorView2fD& A, const TensorView2fD& x) override;

        void init(const TensorView2dD& A) override;
        void factorize(const TensorView2dD& A) override;
        void solve(const TensorView2dD& A, const TensorView2dD& x) override;
    private:
        Tensor<i32, 1, Location::Device> ipiv{m_memory.get()};
        Tensor<f32, 1, Location::Device> buffer{m_memory.get()};
        Tensor<i32, 1, Location::Device> info_d{m_memory.get()}; // single value
        Tensor<i32, 1, Location::Host> info_h{m_memory.get()}; // single value
};

std::unique_ptr<LU> BackendCUDA::create_lu_solver() { return std::make_unique<CUDA_LU>(create_memory_manager()); }

CUDA_LU::CUDA_LU(std::unique_ptr<Memory> memory) : LU(std::move(memory)) {}

void CUDA_LU::init(const TensorView2fD& A) {
    int bufsize = 0;
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(CUSolverCtx::get().dn_handle(), A.shape(0), A.shape(1), A.ptr(), A.stride(1), &bufsize));
    info_d.init({1});
    info_h.init({1});
    buffer.init({bufsize});
    ipiv.init({std::min(A.shape(0), A.shape(1))});
}

void CUDA_LU::factorize(const TensorView2fD& A) {
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

void CUDA_LU::solve(const TensorView2fD& A, const TensorView2fD& x) {
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

class CUDA_LSQ final : public LSQ {
    public:
        explicit CUDA_LSQ(std::unique_ptr<Memory> memory) : LSQ(std::move(memory)) {}
        virtual ~CUDA_LSQ() {
            m_memory->free(Location::Device, m_workspace);
        };
        
        void init(const TensorView2fD& A, const TensorView2fD& B) override;
        void solve(const TensorView2fD& A, const TensorView2fD& B) override;

        void init(const TensorView2dD& A, const TensorView2dD& B) override {};
        void solve(const TensorView2dD& A, const TensorView2dD& B) override {};

    protected:
        Tensor2fD m_X{m_memory.get()};
        Tensor<i32, 1, Location::Host> m_niters{m_memory.get()}; // scalar
        Tensor<i32, 1, Location::Device> m_info{m_memory.get()}; // scalar

        void* m_workspace = nullptr;
        size_t m_workspace_sizebytes = 0; 
};

std::unique_ptr<LSQ> BackendCUDA::create_lsq_solver() { return std::make_unique<CUDA_LSQ>(create_memory_manager()); }

void CUDA_LSQ::init(
    const TensorView2fD& A,
    const TensorView2fD& B
)
{
    m_X.init({B.shape(0), B.shape(1)});
    m_niters.init({1});
    m_info.init({1});
    CHECK_CUSOLVER(cusolverDnSHgels_bufferSize(
        CUSolverCtx::get().dn_handle(),
        A.shape(0),
        A.shape(1),
        B.shape(1),
        A.ptr(),
        A.stride(1),
        B.ptr(),
        B.stride(1),
        m_X.ptr(),
        m_X.view().stride(1),
        m_workspace,
        &m_workspace_sizebytes
    ));
    m_workspace = m_memory->alloc(Location::Device, (i64)m_workspace_sizebytes);
}

void CUDA_LSQ::solve(
    const TensorView2fD& A,
    const TensorView2fD& B
) {
    CHECK_CUSOLVER(cusolverDnSHgels(
        CUSolverCtx::get().dn_handle(),
        A.shape(0),
        A.shape(1),
        B.shape(1),
        A.ptr(),
        A.stride(1),
        B.ptr(),
        B.stride(1),
        m_X.ptr(),
        m_X.view().stride(1),
        m_workspace,
        m_workspace_sizebytes,
        m_niters.ptr(),
        m_info.ptr()
    ));
    if (m_niters.view()(0) > 0) {
        std::printf("CUDA LSQ IRS converged in %d iterations\n", m_niters.view()(0));
    }
    m_info.view().to(m_niters.view());
    if (m_niters.view()(0) != 0) {
        std::printf("CUDA LSQ failed with error code %d\n", m_niters.view()(0));
    }
    m_X.view().to(B);
}
#include "vlm_backend_cpu.hpp"

#include <lapacke.h>

#define CHECK_LAPACK(call) \
    do { \
        const lapack_int err = (call); \
        if (err != 0) { \
            fprintf(stderr, "LAPACKE Error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

using namespace vlm;

class CPU_LU final : public LU {
    public:
        explicit CPU_LU(std::unique_ptr<Memory> memory);
        ~CPU_LU() override = default;
        
        void init(const TensorView2fD& A) override;
        void factorize(const TensorView2fD& A) override;
        void solve(const TensorView2fD& A, const TensorView2fD& x) override;

        void init(const TensorView2dD& A) override;
        void factorize(const TensorView2dD& A) override;
        void solve(const TensorView2dD& A, const TensorView2dD& x) override;

    private:
        Tensor<i32, 1, Location::Device> ipiv{m_memory.get()};
};

CPU_LU::CPU_LU(std::unique_ptr<Memory> memory) : LU(std::move(memory)) {}

void CPU_LU::init(const TensorView2fD& A) {
    ipiv.init({A.shape(0)}); // row pivoting
}

void CPU_LU::factorize(const TensorView2fD& A) {
    assert(ipiv.view().shape(0) == A.shape(0));
    CHECK_LAPACK(LAPACKE_sgetrf(
        LAPACK_COL_MAJOR,
        A.shape(0),
        A.shape(1),
        A.ptr(),
        A.stride(1),
        ipiv.ptr()
    ));
}

void CPU_LU::solve(const TensorView2fD& A, const TensorView2fD& x) {
    CHECK_LAPACK(LAPACKE_sgetrs(
        LAPACK_COL_MAJOR,
        'N',
        A.shape(1),
        x.shape(1),
        A.ptr(),
        A.stride(1),
        ipiv.ptr(),
        x.ptr(),
        x.stride(1)
    ));
}

void CPU_LU::init(const TensorView2dD& A) {
    ipiv.init({A.shape(0)}); // row pivoting
}

void CPU_LU::factorize(const TensorView2dD& A) {
    assert(ipiv.view().shape(0) == A.shape(0));
    CHECK_LAPACK(LAPACKE_dgetrf(
        LAPACK_COL_MAJOR,
        A.shape(0),
        A.shape(1),
        A.ptr(),
        A.stride(1),
        ipiv.ptr()
    ));
}

void CPU_LU::solve(const TensorView2dD& A, const TensorView2dD& x) {
    CHECK_LAPACK(LAPACKE_dgetrs(
        LAPACK_COL_MAJOR,
        'N',
        A.shape(1),
        x.shape(1),
        A.ptr(),
        A.stride(1),
        ipiv.ptr(),
        x.ptr(),
        x.stride(1)
    ));
}

class CPU_LSQ final : public LSQ {
    public:
        explicit CPU_LSQ(std::unique_ptr<Memory> memory) : LSQ(std::move(memory)) {}
        virtual ~CPU_LSQ() = default;
        
        void init(
            const TensorView2fD& A,
            const TensorView2fD& B
        ) override;
        void solve(
            const TensorView2fD& A,
            const TensorView2fD& B
        ) override;
        void init(
            const TensorView2dD& A,
            const TensorView2dD& B
        ) override;
        void solve(
            const TensorView2dD& A,
            const TensorView2dD& B
        ) override;

    protected:
        Tensor1fD m_workspace{m_memory.get()};
        Tensor<f64, 1, Location::Device> m_workspace_d{m_memory.get()};
};

void CPU_LSQ::init(
    const TensorView2fD& A,
    const TensorView2fD& B
) {
    f32 work_size;
    CHECK_LAPACK(LAPACKE_sgels_work(
        LAPACK_COL_MAJOR,
        'N',
        A.shape(0),
        A.shape(1),
        B.shape(1),
        A.ptr(),
        A.stride(1),
        B.ptr(),
        B.stride(1),
        &work_size,
        -1
    ));
    m_workspace.init({static_cast<i64>(work_size)});
}

void CPU_LSQ::solve(
    const TensorView2fD& A,
    const TensorView2fD& B
) {
    CHECK_LAPACK(
        LAPACKE_sgels_work(
            LAPACK_COL_MAJOR,
            'N',
            A.shape(0),
            A.shape(1),
            B.shape(1),
            A.ptr(),
            A.stride(1),
            B.ptr(),
            B.stride(1),
            m_workspace.ptr(),
            m_workspace.size()
        )
    );
}

void CPU_LSQ::init(
    const TensorView2dD& A,
    const TensorView2dD& B
) {
    f64 work_size;
    CHECK_LAPACK(LAPACKE_dgels_work(
        LAPACK_COL_MAJOR,
        'N',
        A.shape(0),
        A.shape(1),
        B.shape(1),
        A.ptr(),
        A.stride(1),
        B.ptr(),
        B.stride(1),
        &work_size,
        -1
    ));
    m_workspace_d.init({static_cast<i64>(work_size)});
}

void CPU_LSQ::solve(
    const TensorView2dD& A,
    const TensorView2dD& B
) {
    CHECK_LAPACK(
        LAPACKE_dgels_work(
            LAPACK_COL_MAJOR,
            'N',
            A.shape(0),
            A.shape(1),
            B.shape(1),
            A.ptr(),
            A.stride(1),
            B.ptr(),
            B.stride(1),
            m_workspace_d.ptr(),
            m_workspace_d.size()
        )
    );
}

std::unique_ptr<LU> BackendCPU::create_lu_solver() { return std::make_unique<CPU_LU>(create_memory_manager()); }
std::unique_ptr<LSQ> BackendCPU::create_lsq_solver() { return std::make_unique<CPU_LSQ>(create_memory_manager()); }

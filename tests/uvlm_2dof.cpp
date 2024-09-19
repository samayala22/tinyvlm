#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"

#include "tinycombination.hpp"

using namespace vlm;

class NewmarkBeta {
    public:
        NewmarkBeta(std::unique_ptr<Memory> memory, const f32 beta = 0.25f, const f32 gamma = 0.5f) : m_memory(std::move(memory)), m_beta(beta), m_gamma(gamma) {};
        ~NewmarkBeta() = default;

        void run(View<f32, Tensor<2>>& M, View<f32, Tensor<2>>& C, View<f32, Tensor<2>>& K, View<f32, Tensor<1>>& F, View<f32, Tensor<1>>& u0, View<f32, Tensor<1>>& v0, View<f32, Tensor<1>>& t);

    private:
        std::unique_ptr<BLAS> m_blas;
        std::unique_ptr<Memory> m_memory;
        std::unique_ptr<LUSolver> m_solver;
        const f32 m_beta;
        const f32 m_gamma;

        Buffer<f32, MemoryLocation::Device, Tensor<2>> K_eff{*m_memory}; // effective stiffness
        Buffer<f32, MemoryLocation::Device, Tensor<1>> a0{*m_memory}; // initial acceleration
        Buffer<f32, MemoryLocation::Device, Tensor<1>> du{*m_memory}; // incremental displacement
        Buffer<f32, MemoryLocation::Device, Tensor<1>> factor{*m_memory}; // intermediary vector
        Buffer<f32, MemoryLocation::Device, Tensor<2>> u{*m_memory}; // dof x tsteps position history
        Buffer<f32, MemoryLocation::Device, Tensor<2>> v{*m_memory}; // dof x tsteps velocity history
        Buffer<f32, MemoryLocation::Device, Tensor<2>> a{*m_memory}; // dof x tsteps acceleration history
};

void NewmarkBeta::run(View<f32, Tensor<2>>& M, View<f32, Tensor<2>>& C, View<f32, Tensor<2>>& K, View<f32, Tensor<1>>& F, View<f32, Tensor<1>>& u0, View<f32, Tensor<1>>& v0, View<f32, Tensor<1>>& t) {
    assert(M.layout.shape(0) == C.layout.shape(0));
    assert(M.layout.shape(1) == C.layout.shape(1));
    assert(C.layout.shape(0) == K.layout.shape(0));
    assert(C.layout.shape(1) == K.layout.shape(1));
    assert(K.layout.shape(0) == F.layout.shape(0));
    assert(u0.layout.shape(0) == F.layout.shape(0));
    assert(v0.layout.shape(0) == F.layout.shape(0));

    const Tensor<2> time_series{{F.layout.shape(0), t.layout.shape(0)}}; // dofs x timesteps

    // TODO: preallocate a number of timesteps and only resize when necessary
    u.dealloc();
    v.dealloc();
    a.dealloc();

    u.alloc(time_series);
    v.alloc(time_series);
    a.alloc(time_series);

    m_memory->copy(MemoryTransfer::DeviceToDevice, u.d_view().ptr, u0.ptr, u0.size_bytes());
    m_memory->copy(MemoryTransfer::DeviceToDevice, v.d_view().ptr, v0.ptr, v0.size_bytes());
    
    // a[:,0] = np.linalg.solve(M, F[0] - C @ v0 - K @ x0)
    View<f32, Tensor<1>> a0_col0 = a.d_view().layout.slice(a.d_view().ptr, all, 0);
    m_memory->copy(MemoryTransfer::DeviceToDevice, a0_col0.ptr, F.ptr, F.size_bytes());
    m_blas->gemv(-1.0f, C, v0, 1.0f, a0_col0);
    m_blas->gemv(-1.0f, K, u0, 1.0f, a0_col0);
    // solver->factorize(M);
    // solver->solve(M, a0_col0);

    const f32 dt = t[1] - t[0];
    const f32 x2 = 1;
    const f32 x1 = m_gamma / (m_beta * dt);
    const f32 x0 = 1 / (m_beta * dt*dt);
    const f32 xd0 = 1 / (m_beta * dt);
    const f32 xd1 = m_gamma / m_beta;
    const f32 xdd0 = 1/(2*m_beta);
    const f32 xdd1 = - dt * (1 - m_gamma / (2*m_beta));

    // K_eff = K + a0 * M + a1 * C
    // This whole thing should be fused into a single kernel...
    m_memory->copy(MemoryTransfer::DeviceToDevice, K_eff.d_view().ptr, K.ptr, K.size_bytes());
    // m_blas->axpy(x0, M, K_eff);
    // m_blas->axpy(x1, C, K_eff);
}

int main() {
    const u64 ni = 20;
    const u64 nj = 5;
    // vlm::Executor::instance(1);
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_" + std::to_string(ni) + "x" + std::to_string(nj) + ".x"};
    const std::vector<std::string> backends = {"cpu"};

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 10.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 0.5; // reduced frequency
    const f32 omega = k * 2.0f * u_inf / (2*b);
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods
    //const f32 t_final = 5.0f;

    Kinematics kinematics{};

    const f32 initial_angle = 0.0f;

    const auto initial_pose = rotation_matrix(
        linalg::alias::float3{0.0f, 0.0f, 0.0f}, // take into account quarter chord panel offset
        linalg::alias::float3{0.0f, 1.0f, 0.0f},
        to_radians(initial_angle)
    );
    
    // Sudden acceleration
    const f32 alpha = to_radians(5.0f);
    kinematics.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            -u_inf*std::cos(alpha)*t,
            0.0f,
            -u_inf*std::sin(alpha)*t
        });
    });

    for (const auto& [mesh_name, backend_name] : solvers) {
        UVLM simulation{backend_name, {mesh_name}};
        simulation.run({kinematics}, {initial_pose}, t_final);
    }
    return 0;
}
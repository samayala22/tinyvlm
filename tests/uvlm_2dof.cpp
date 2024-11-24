#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "vlm_kinematics.hpp"
#include "vlm_integrator.hpp"

#include "tinycombination.hpp"
#include "tinypbar.hpp"

#include <tuple>
#include <fstream>

using namespace vlm;

constexpr i32 DOF = 2;

class UVLM_2DOF final: public Simulation {
    public:
        UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~UVLM_2DOF() = default;
        void run(const Assembly& assembly, f32 t_final);
    
        // Structure
        Tensor2D<Location::Host> M_h{backend->memory.get()};
        Tensor2D<Location::Host> C_h{backend->memory.get()};
        Tensor2D<Location::Host> K_h{backend->memory.get()};
        Tensor2D<Location::Host> u_h{backend->memory.get()};
        Tensor2D<Location::Host> v_h{backend->memory.get()};
        Tensor2D<Location::Host> a_h{backend->memory.get()};
        Tensor1D<Location::Host> t_h{backend->memory.get()};

        Tensor2D<Location::Device> M_d{backend->memory.get()};
        Tensor2D<Location::Device> C_d{backend->memory.get()};
        Tensor2D<Location::Device> K_d{backend->memory.get()};
        Tensor2D<Location::Device> u_d{backend->memory.get()}; // dof x tsteps
        Tensor2D<Location::Device> v_d{backend->memory.get()}; // dof x tsteps
        Tensor2D<Location::Device> a_d{backend->memory.get()}; // dof x tsteps
        Tensor1D<Location::Device> du{backend->memory.get()}; // dof
        Tensor1D<Location::Device> dv{backend->memory.get()}; // dof
        Tensor1D<Location::Device> da{backend->memory.get()}; // dof

        NewmarkBeta integrator{backend.get()};

        std::unique_ptr<LU> solver;
    private:
        void alloc_buffers();
};

UVLM_2DOF::UVLM_2DOF(const std::string& backend_name, const std::vector<std::string>& meshes) : Simulation(backend_name, meshes) {
    solver = backend->create_lu_solver();
    alloc_buffers();
}

void UVLM_2DOF::alloc_buffers() {
    // Host tensors
    M_h.init({DOF,DOF});
    C_h.init({DOF,DOF});
    K_h.init({DOF,DOF});

    // Device tensors
    M_d.init({DOF,DOF});
    C_d.init({DOF,DOF});
    K_d.init({DOF,DOF});
    du.init({DOF});
    dv.init({DOF});
    da.init({DOF});
}

void UVLM_2DOF::run(const Assembly& assembly, f32 t_final) {
    return;
}

int main() {
    const i64 ni = 20;
    const i64 nj = 5;
    // vlm::Executor::instance(1);
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_" + std::to_string(ni) + "x" + std::to_string(nj) + ".x"};
    // const std::vector<std::string> meshes = {"../../../../mesh/rectangular_4x4.x"};
    
    const std::vector<std::string> backends = {"cpu"};

    auto simulations = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 10.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 0.5; // reduced frequency
    const f32 omega = k * 2.0f * u_inf / (2*b);
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods

    KinematicsTree kinematics_tree;
    
    // Sudden acceleration
    const f32 alpha = to_radians(5.0f);
    auto freestream = kinematics_tree.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({
            -u_inf*std::cos(alpha)*t,
            0.0f,
            -u_inf*std::sin(alpha)*t
        });
    });

    for (const auto& [mesh_name, backend_name] : simulations) {
        Assembly assembly(freestream);
        assembly.add(mesh_name, freestream);
        UVLM simulation{backend_name, {mesh_name}};
        simulation.run(assembly, t_final);
    }
    return 0;
}
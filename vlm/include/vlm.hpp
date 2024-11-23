#pragma once

#include <memory>

#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_backend.hpp"
#include "tinyfwd.hpp"

namespace vlm {

class Simulation {
    public:
        std::unique_ptr<Backend> backend;
        // std::unique_ptr<Backend> backend_cpu; // TEMPORARY

        // Common geometry buffers
        MultiTensor3D<Location::Device> verts_wing_init{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3D<Location::Host> verts_wing_init_h{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3D<Location::Device> verts_wing{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3D<Location::Host> verts_wing_h{backend->memory.get()}; // (nc+1)*(ns+1)*3
        MultiTensor3D<Location::Device> verts_wake{backend->memory.get()}; // (nw+1)*(ns+1)*3

        MultiDim<2> assembly_wings;

        Simulation(const std::string& backend_name, const std::vector<std::string>& meshes);
        virtual ~Simulation() = default;
};

class VLM final: public Simulation {
    public:
        VLM(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~VLM() = default;
        AeroCoefficients run(const FlowData& flow);

        std::vector<i32> condition0; // TODO: remove

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        MultiTensor2D<Location::Device> areas_d{backend->memory.get()};
        
        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        MultiTensor2D<Location::Device> gamma_wing{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wake{backend->memory.get()}; // nw*ns
        MultiTensor2D<Location::Device> gamma_wing_prev{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wing_delta{backend->memory.get()}; // nc*ns

        MultiTensor3D<Location::Device> local_velocities{backend->memory.get()}; // ns*nc*3
        Tensor2D<Location::Host> wake_transform{backend->memory.get()};
        Tensor3D<Location::Device> transforms{backend->memory.get()};

        std::unique_ptr<LU> solver;
    private:
        void alloc_buffers();
};

class NLVLM final: public Simulation {
    public:

        static constexpr f64 DEFAULT_TOL = 1e-5;
        static constexpr i64 DEFAULT_MAX_ITER = 100;

        NLVLM(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~NLVLM() = default;
        AeroCoefficients run(const FlowData& flow, const Database& db);

        const i64 max_iter = DEFAULT_MAX_ITER;
        const f64 tol = DEFAULT_TOL;
        std::vector<i32> condition0;

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        MultiTensor2D<Location::Device> areas_d{backend->memory.get()};

        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        MultiTensor2D<Location::Device> gamma_wing{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wake{backend->memory.get()}; // nw*ns
        MultiTensor2D<Location::Device> gamma_wing_prev{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wing_delta{backend->memory.get()}; // nc*ns

        MultiTensor3D<Location::Device> local_velocities{backend->memory.get()}; // ns*nc*3
        MultiTensor1D<Location::Host> strip_alphas{backend->memory.get()}; // ns
        Tensor2D<Location::Host> wake_transform{backend->memory.get()};
        Tensor3D<Location::Device> transforms{backend->memory.get()};

        std::unique_ptr<LU> solver;
    private:
        void alloc_buffers();
};

class UVLM final: public Simulation {
    public:
        UVLM(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~UVLM() = default;
        void run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::float4x4>& initial_pose, f32 t_final);

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Host> colloc_h{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};
        MultiTensor2D<Location::Device> areas_d{backend->memory.get()};

        MultiTensor3D<Location::Device> verts_wing_pos{backend->memory.get()}; // (nc+1)*(ns+1)*3

        // Data
        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        MultiTensor2D<Location::Device> gamma_wing{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wake{backend->memory.get()}; // nw*ns
        MultiTensor2D<Location::Device> gamma_wing_prev{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wing_delta{backend->memory.get()}; // nc*ns

        MultiTensor3D<Location::Device> velocities{backend->memory.get()}; // ns*nc*3
        MultiTensor3D<Location::Host> velocities_h{backend->memory.get()}; // ns*nc*3

        Tensor3D<Location::Host> transforms_h{backend->memory.get()};
        Tensor3D<Location::Device> transforms{backend->memory.get()};
        
        Tensor1D<Location::Host> t_h{backend->memory.get()};
        std::unique_ptr<LU> solver;

        std::vector<i32> condition0;
    private:
        void alloc_buffers();
};

} // namespace vlm

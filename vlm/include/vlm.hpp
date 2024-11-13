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
        i32 nb_meshes;
        // Common geometry buffers
        Mesh mesh{*backend->memory};
        // Dimensions of the buffers
        std::vector<SurfaceDims> wing_panels;
        std::vector<SurfaceDims> wing_vertices;
        std::vector<SurfaceDims> wake_panels;
        std::vector<SurfaceDims> wake_vertices;

        MultiDim<2> assembly_wings;
        
        // Misc
        std::vector<linalg::alias::float4x4> wing_positions; // todo: move this somewhere

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
        
        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        MultiTensor2D<Location::Device> gamma_wing{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wake{backend->memory.get()}; // nw*ns
        MultiTensor2D<Location::Device> gamma_wing_prev{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wing_delta{backend->memory.get()}; // nc*ns

        Buffer<f32, Location::HostDevice, MultiSurface> local_velocities{*backend->memory}; // ns*nc*3
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

        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        MultiTensor2D<Location::Device> gamma_wing{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wake{backend->memory.get()}; // nw*ns
        MultiTensor2D<Location::Device> gamma_wing_prev{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wing_delta{backend->memory.get()}; // nc*ns

        Buffer<f32, Location::HostDevice, MultiSurface> local_velocities{*backend->memory}; // ns*nc*3
        Buffer<f32, Location::Host, MultiSurface> strip_alphas{*backend->memory}; // ns
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
        void run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::alias::float4x4>& initial_pose, f32 t_final);

        MultiTensor3D<Location::Device> colloc_d{backend->memory.get()};
        MultiTensor3D<Location::Device> normals_d{backend->memory.get()};

        // Mesh
        Buffer<f32, Location::HostDevice, MultiSurface> verts_wing_pos{*backend->memory}; // (nc+1)*(ns+1)*3
        Buffer<f32, Location::Host, MultiSurface> colloc_pos{*backend->memory}; // (nc)*(ns)*3

        // Data
        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        MultiTensor2D<Location::Device> gamma_wing{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wake{backend->memory.get()}; // nw*ns
        MultiTensor2D<Location::Device> gamma_wing_prev{backend->memory.get()}; // nc*ns
        MultiTensor2D<Location::Device> gamma_wing_delta{backend->memory.get()}; // nc*ns

        Buffer<f32, Location::HostDevice, MultiSurface> velocities{*backend->memory}; // ns*nc*3
        Tensor3D<Location::Host> transforms_h{backend->memory.get()};
        Tensor3D<Location::Device> transforms{backend->memory.get()};
        
        std::unique_ptr<LU> solver;

        std::vector<f32> vec_t; // timesteps
        std::vector<f32> local_dt; // per mesh dt (pre reduction)
        std::vector<i32> condition0;

    private:
        void alloc_buffers();
};

} // namespace vlm

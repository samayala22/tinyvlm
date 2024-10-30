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
        u32 nb_meshes;
        // Common geometry buffers
        Mesh mesh{*backend->memory};
        // Dimensions of the buffers
        std::vector<SurfaceDims> wing_panels;
        std::vector<SurfaceDims> wing_vertices;
        std::vector<SurfaceDims> wake_panels;
        std::vector<SurfaceDims> wake_vertices;
        
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

        std::vector<u32> condition0;
        
        // Pulic for output purposes (eg dump gamma to file)
        Buffer<f32, Location::Device, Matrix<MatrixLayout::ColMajor>> lhs{*backend->memory}; // (ns*nc)^2
        Buffer<f32, Location::Device, MultiSurface> rhs{*backend->memory}; // ns*nc
        Buffer<f32, Location::HostDevice, MultiSurface> gamma_wing{*backend->memory}; // nc*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wake{*backend->memory}; // nw*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wing_prev{*backend->memory}; // nc*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wing_delta{*backend->memory}; // nc*ns
        Buffer<f32, Location::Device, MultiSurface> local_velocities{*backend->memory}; // ns*nc*3
        Buffer<f32, Location::HostDevice, Tensor<3>> transforms{*backend->memory};
    private:
        void alloc_buffers();
};

class NLVLM final: public Simulation {
    public:

        static constexpr f64 DEFAULT_TOL = 1e-5;
        static constexpr u64 DEFAULT_MAX_ITER = 100;

        NLVLM(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~NLVLM() = default;
        AeroCoefficients run(const FlowData& flow, const Database& db);

        const u64 max_iter = DEFAULT_MAX_ITER;
        const f64 tol = DEFAULT_TOL;
        std::vector<u32> condition0;

        Buffer<f32, Location::Device, Matrix<MatrixLayout::ColMajor>> lhs; // (ns*nc)^2
        Buffer<f32, Location::Device, MultiSurface> rhs; // ns*nc
        Buffer<f32, Location::HostDevice, MultiSurface> gamma_wing; // nc*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wake; // nw*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wing_prev; // nc*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wing_delta; // nc*ns
        Buffer<f32, Location::HostDevice, MultiSurface> local_velocities; // ns*nc*3
        Buffer<f32, Location::Host, MultiSurface> strip_alphas; // ns
        Buffer<f32, Location::HostDevice, Tensor<3>> transforms;

    private:
        void alloc_buffers();
};

class UVLM final: public Simulation {
    public:
        UVLM(const std::string& backend_name, const std::vector<std::string>& meshes);
        ~UVLM() = default;
        void run(const std::vector<Kinematics>& kinematics, const std::vector<linalg::alias::float4x4>& initial_pose, f32 t_final);
    
        // Mesh
        Buffer<f32, Location::HostDevice, MultiSurface> verts_wing_pos{*backend->memory}; // (nc+1)*(ns+1)*3
        Buffer<f32, Location::Host, MultiSurface> colloc_pos{*backend->memory}; // (nc)*(ns)*3

        // Data
        Buffer<f32, Location::Device, Matrix<MatrixLayout::ColMajor>> lhs{*backend->memory}; // (ns*nc)^2
        Buffer<f32, Location::Device, MultiSurface> rhs{*backend->memory}; // ns*nc
        Buffer<f32, Location::HostDevice, MultiSurface> gamma_wing{*backend->memory}; // nc*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wake{*backend->memory}; // nw*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wing_prev{*backend->memory}; // nc*ns
        Buffer<f32, Location::Device, MultiSurface> gamma_wing_delta{*backend->memory}; // nc*ns
        Buffer<f32, Location::HostDevice, MultiSurface> velocities{*backend->memory}; // ns*nc*3
        Buffer<f32, Location::HostDevice, Tensor<3>> transforms{*backend->memory}; // 4*4*nb_meshes
        
        std::vector<f32> vec_t; // timesteps
        std::vector<f32> local_dt; // per mesh dt (pre reduction)
        std::vector<u32> condition0;

    private:
        void alloc_buffers();
};

} // namespace vlm

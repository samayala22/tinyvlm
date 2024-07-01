#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_allocator.hpp"

namespace vlm {

class Backend {
    public:
        Allocator allocator;
        
        // Constant inital position meshes (for kinematics)
        MeshGeom* hh_mesh_geom; // borrowed ptr
        MeshGeom* hd_mesh_geom; 
        MeshGeom* dd_mesh_geom; 

        // Mutable meshes (temporal state)
        // Mesh2* hh_mesh; // host ptr to host buffers for io
        Mesh2* hd_mesh; // host ptr to device buffers
        Mesh2* dd_mesh; // device ptr to device buffers for kernels

        Data* hd_data;
        Data* dd_data;

        i32* d_solver_info = nullptr;
        i32* d_solver_ipiv = nullptr;
        f32* d_solver_buffer = nullptr;

        f32 sigma_vatistas = 0.0f;
        Backend() = default;
        ~Backend();
        void init(MeshGeom* mesh_geom, u64 timesteps); // Acts as delayed constructor
        virtual void reset() = 0;
        virtual void lhs_assemble() = 0;
        virtual void compute_rhs() = 0;
        virtual void add_wake_influence() = 0;
        virtual void wake_rollup(float dt) = 0;
        virtual void shed_gamma() = 0;
        virtual void lu_factor() = 0;
        virtual void lu_solve() = 0;
        virtual f32 compute_coefficient_cl(const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;
        virtual f32 compute_coefficient_unsteady_cl(const linalg::alias::float3& freestream, const SoA_3D_t<f32>& vel, f32 dt, const f32 area, const u64 j, const u64 n) = 0;
        f32 compute_coefficient_cl(const FlowData& flow);
        virtual linalg::alias::float3 compute_coefficient_cm(const FlowData& flow, const f32 area, const f32 chord, const u64 j, const u64 n) = 0;
        linalg::alias::float3 compute_coefficient_cm(const FlowData& flow);
        virtual f32 compute_coefficient_cd(const FlowData& flow, const f32 area, const u64 j, const u64 n) = 0;
        f32 compute_coefficient_cd(const FlowData& flow);
        virtual void compute_delta_gamma() = 0;
        virtual void set_velocities(const linalg::alias::float3& vel) = 0;
        virtual void set_velocities(const SoA_3D_t<f32>& vels) = 0;

        virtual void mesh_metrics(const f32 alpha) = 0;
        virtual void mesh_move(const linalg::alias::float4x4& transform) = 0;
        virtual void update_wake(const linalg::alias::float3& freestream) = 0; // TEMPORARY
    private:
        virtual f32 mesh_mac(u64 j, u64 n) = 0; // mean chord
        virtual f32 mesh_area(const u64 i, const u64 j, const u64 m, const u64 n) = 0; // mean span
};

std::unique_ptr<Backend> create_backend(const std::string& backend_name, MeshGeom* mesh, int timesteps);
std::vector<std::string> get_available_backends();

} // namespace vlm

#include <fstream>

#include "vlm.hpp"
#include "vlm_utils.hpp"
#include "tinycombination.hpp"

#define ASSERT_EQ(x, y) \
    do { \
        auto val1 = (x); \
        auto val2 = (y); \
        if (!(val1 == val2)) { \
            std::cerr << "Assertion failed: " << #x << " == " << #y << " (Left: " << val1 << ", Right: " << val2 << ")\n"; \
            std::abort(); \
        } \
    } while (0)

#define ASSERT_NEAR(x, y, tol) \
    do { \
        auto val1 = (x); \
        auto val2 = (y); \
        if (!(std::abs(val1 - val2) <= tol)) { \
            std::cerr << "Assertion failed: |" << #x << " - " << #y << "| <= " << tol << " (Left: " << val1 << ", Right: " << val2 << ", Diff: " << std::abs(val1 - val2) << ")\n"; \
            std::abort(); \
        } \
    } while (0)

using namespace vlm;
using namespace linalg::ostream_overloads;

class PM final: public Simulation {
    public:
        PM(const std::string& backend_name, const std::string& mesh);
        ~PM() = default;

        // Geometry
        MultiTensor3D<Location::Device> colloc{backend->memory.get()};
        MultiTensor3D<Location::Device> normals{backend->memory.get()};
        MultiTensor2D<Location::Device> areas{backend->memory.get()};

        // Data
        MultiTensor2D<Location::Device> mu{backend->memory.get()};
        MultiTensor2D<Location::Device> mu_prev{backend->memory.get()};
        MultiTensor3D<Location::Device> surf_vel{backend->memory.get()};
        MultiTensor2D<Location::Device> cp{backend->memory.get()};

        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        
        std::unique_ptr<LU> mu_solver;

        void run();
    private:
        void alloc_buffers();
        void rhs_assemble(
            TensorView1D<Location::Device>& rhs,
            const MultiTensorView3D<Location::Device>& colloc,
            const MultiTensorView3D<Location::Device>& normals,
            const MultiTensorView3D<Location::Device>& verts_wing,
            const linalg::float3& freestream
        );

        void surface_velocities(
            TensorView3D<Location::Device>& surface_velocities,
            const TensorView3D<Location::Device>& verts_wing,
            const TensorView2D<Location::Device>& areas,
            const TensorView2D<Location::Device>& mu,
            const linalg::float3& freestream
        );
        void pressure_coefficient(
            TensorView3D<Location::Device>& surface_velocities,
            TensorView2D<Location::Device>& cp
        );
};

PM::PM(const std::string& backend_name, const std::string& mesh) : Simulation(backend_name, {mesh}, false) {
    mu_solver = backend->create_lu_solver();
    alloc_buffers();
}

inline i64 total_panels(const MultiDim<2>& assembly_wing) {
    i64 total = 0;
    for (const auto& wing : assembly_wing) {
        total += wing[0] * wing[1];
    }
    return total;
}

void PM::alloc_buffers() {
    const i64 n = total_panels(assembly_wings);
    MultiDim<3> panels_3D;
    MultiDim<2> panels_2D;
    MultiDim<3> verts_wing_3D;

    for (const auto& [ns, nc] : assembly_wings) {
        panels_3D.push_back({ns, nc, 3});
        panels_2D.push_back({ns, nc});
        verts_wing_3D.push_back({ns+1, nc+1, 4});
    }

    normals.init(panels_3D);
    colloc.init(panels_3D);
    areas.init(panels_2D);

    mu.init(panels_2D);
    mu_prev.init(panels_2D);
    surf_vel.init(panels_3D); // should actually be ns*nc*2 tensor
    cp.init(panels_2D);

    lhs.init({n, n});
    rhs.init({n});

    mu_solver->init(lhs.view());
}

// Global to local transformation matrix
linalg::float4x4 panel_local_frame(
    const linalg::float3& v0,
    const linalg::float3& v1,
    const linalg::float3& v2,
    const linalg::float3& v3
)
{
    const linalg::float3 k = linalg::normalize(linalg::cross(v3 - v1, v2 - v0));
    const linalg::float3 j = linalg::normalize(v1 - v0);
    const linalg::float3 i = linalg::cross(j, k);
    const linalg::float3 center = 0.25f * (v0 + v1 + v2 + v3);
    return linalg::inverse(linalg::transpose(linalg::float4x4{
        {i.x, j.x, k.x, center.x},
        {i.y, j.y, k.y, center.y},
        {i.z, j.z, k.z, center.z},
        {0.0f, 0.0f, 0.0f, 1.0f}
    })); // transpose because construction is done in col major
}

f32 doublet_edge_influence(
    const linalg::float3& v1,
    const linalg::float3& v2,
    const linalg::float3& x
) 
{
    f32 influence = 0.0f;
    const f32 r1 = linalg::length(x - v1);
    const f32 r2 = linalg::length(x - v2);
    const f32 e1 = pow(x.x - v1.x, 2) + pow(x.z, 2);
    const f32 e2 = pow(x.x - v2.x, 2) + pow(x.z, 2);
    const f32 h1 = (x.x - v1.x)*(x.y - v1.y);
    const f32 h2 = (x.x - v2.x)*(x.y - v2.y);
    const f32 m = (v2.y - v1.y) / (v2.x - v1.x);
    const f32 F = (m*e1 - h1) / (x.z*r1);
    const f32 G = (m*e2 - h2) / (x.z*r2);
    if (std::abs(F - G) > EPS_f) {
        influence = std::atan2(F-G, 1+F*G);
    }
    return influence;
}

f32 source_edge_influence(
    const linalg::float3& v1,
    const linalg::float3& v2,
    const linalg::float3& x
)
{
    f32 influence = 0.0f;
    const f32 r1 = linalg::length(x - v1);
    const f32 r2 = linalg::length(x - v2);
    const f32 d12 = linalg::length(v2 - v1);

    if (d12 > EPS_f || (r1+r2-d12) > EPS_f) {
        influence = ((x.x-v1.x)*(v2.y - v1.y) - (x.y-v1.y)*(v2.x-v1.x))
                  / d12 * std::log((r1+r2+d12) / (r1+r2-d12));
    }

    if (std::fabsf(x.z) > EPS_f) {
        influence -= x.z * doublet_edge_influence(v1, v2, x);
    }
    return influence;
}

inline linalg::float3 to_float3(const linalg::float4& v) {
    return {v.x, v.y, v.z};
}

inline linalg::float4 to_float4(const linalg::float3& v, f32 w = 1.0f) {
    return {v.x, v.y, v.z, w};
}

// lhs is column major matrix, so we fill column by column
void lhs_assemble(
    TensorView2D<Location::Device>& lhs,
    const MultiTensorView3D<Location::Device>& colloc, // local coords
    const MultiTensorView3D<Location::Device>& verts_wing
) {
    auto& colloc_m = colloc[0];
    auto& verts_wing_m = verts_wing[0];
    for (i64 j = 0; j < colloc_m.shape(1); j++) { // influencer
        for (i64 i = 0; i < colloc_m.shape(0); i++) { // influencer
            const linalg::float3 v0{verts_wing_m(i, j, 0), verts_wing_m(i, j, 1), verts_wing_m(i, j, 2)}; // upper left
            const linalg::float3 v1{verts_wing_m(i+1, j, 0), verts_wing_m(i+1, j, 1), verts_wing_m(i+1, j, 2)}; // upper right
            const linalg::float3 v2{verts_wing_m(i+1, j+1, 0), verts_wing_m(i+1, j+1, 1), verts_wing_m(i+1, j+1, 2)}; // lower right
            const linalg::float3 v3{verts_wing_m(i, j+1, 0), verts_wing_m(i, j+1, 1), verts_wing_m(i, j+1, 2)}; // lower left
            auto g_transform_l = panel_local_frame(v0, v1, v2, v3);
            const linalg::float4 v0_local = linalg::mul(g_transform_l, to_float4(v0));
            const linalg::float4 v1_local = linalg::mul(g_transform_l, to_float4(v1));
            const linalg::float4 v2_local = linalg::mul(g_transform_l, to_float4(v2));
            const linalg::float4 v3_local = linalg::mul(g_transform_l, to_float4(v3));
            
            const i64 influencer_lidx = j * colloc_m.shape(0) + i;

            for (i64 jj = 0; jj < colloc_m.shape(1); jj++) { // influenced
                for (i64 ii = 0; ii < colloc_m.shape(0); ii++) { // influenced
                    const linalg::float4 colloc_inf{colloc_m(ii, jj, 0), colloc_m(ii, jj, 1), colloc_m(ii, jj, 2), 1};
                    const linalg::float4 colloc_inf_local = linalg::mul(g_transform_l, colloc_inf);
                    const i64 influenced_lidx = jj * colloc_m.shape(0) + ii;

                    f32 influence = 0.0f;
                    influence += doublet_edge_influence(to_float3(v0_local), to_float3(v1_local), to_float3(colloc_inf_local));
                    influence += doublet_edge_influence(to_float3(v1_local), to_float3(v2_local), to_float3(colloc_inf_local));
                    influence += doublet_edge_influence(to_float3(v2_local), to_float3(v3_local), to_float3(colloc_inf_local));
                    influence += doublet_edge_influence(to_float3(v3_local), to_float3(v0_local), to_float3(colloc_inf_local));
                    influence /= 4.f * PI_f;
                    if (influenced_lidx == influencer_lidx) {
                        influence = -0.5f;
                    }
                    lhs(influenced_lidx, influencer_lidx) = influence;
                }
            }
        }
    }
}

// Performs the matrix-free assmembly of the rhs with the source influence
void PM::rhs_assemble(
    TensorView1D<Location::Device>& rhs,
    const MultiTensorView3D<Location::Device>& colloc, // local coords
    const MultiTensorView3D<Location::Device>& normals,
    const MultiTensorView3D<Location::Device>& verts_wing,
    const linalg::float3& freestream
)
{
    auto& colloc_m = colloc[0];
    auto& normals_m = normals[0];
    auto& verts_wing_m = verts_wing[0];

    for (i64 j = 0; j < colloc_m.shape(1); j++) { // row
        for (i64 i = 0; i < colloc_m.shape(0); i++) { // row
            const i64 row_idx = j * colloc_m.shape(0) + i;
            const linalg::float4 colloc_inf{colloc_m(i, j, 0), colloc_m(i, j, 1), colloc_m(i, j, 2), 1.0f};

            f32 accumulated_influence = 0.0f;
            for (i64 jj = 0; jj < colloc_m.shape(1); jj++) { // col
                for (i64 ii = 0; ii < colloc_m.shape(0); ii++) { // col
                    const i64 col_idx = jj * colloc_m.shape(0) + ii;
                    const linalg::float3 normal{normals_m(ii, jj, 0), normals_m(ii, jj, 1), normals_m(ii, jj, 2)};
                    // Broadcast
                    const linalg::float3 v0{verts_wing_m(ii, jj, 0), verts_wing_m(ii, jj, 1), verts_wing_m(ii, jj, 2)}; // upper left
                    const linalg::float3 v1{verts_wing_m(ii+1, jj, 0), verts_wing_m(ii+1, jj, 1), verts_wing_m(ii+1, jj, 2)}; // upper right
                    const linalg::float3 v2{verts_wing_m(ii+1, jj+1, 0), verts_wing_m(ii+1, jj+1, 1), verts_wing_m(ii+1, jj+1, 2)}; // lower right
                    const linalg::float3 v3{verts_wing_m(ii, jj+1, 0), verts_wing_m(ii, jj+1, 1), verts_wing_m(ii, jj+1, 2)}; // lower left
                    const auto g_transform_l = panel_local_frame(v0, v1, v2, v3);
                    const linalg::float4 v0_local = linalg::mul(g_transform_l, to_float4(v0));
                    const linalg::float4 v1_local = linalg::mul(g_transform_l, to_float4(v1));
                    const linalg::float4 v2_local = linalg::mul(g_transform_l, to_float4(v2));
                    const linalg::float4 v3_local = linalg::mul(g_transform_l, to_float4(v3));
                    const linalg::float4 colloc_inf_local = linalg::mul(g_transform_l, colloc_inf);
                    const f32 sigma = linalg::dot(normal, freestream);

                    f32 influence = 0.0f;
                    influence += source_edge_influence(to_float3(v0_local), to_float3(v1_local), to_float3(colloc_inf_local));
                    influence += source_edge_influence(to_float3(v1_local), to_float3(v2_local), to_float3(colloc_inf_local));
                    influence += source_edge_influence(to_float3(v2_local), to_float3(v3_local), to_float3(colloc_inf_local));
                    influence += source_edge_influence(to_float3(v3_local), to_float3(v0_local), to_float3(colloc_inf_local));
                    influence *= - 1.0f / (4.f * PI_f); // can move this to the accumulated influence

                    accumulated_influence += sigma * influence;
                }
            }
            rhs(row_idx) = - accumulated_influence;
        }
    }
}

template<typename T>
T clamp(T val, T min, T max) {
    return (val < min) ? min : ((val > max) ? max : val);
}

// Rotate the edge vector 90 degrees counter clockwise
linalg::float2 rot_vec_cc_90(const linalg::float2& v) {
    return {-v.y, v.x};
}

void PM::surface_velocities(
    TensorView3D<Location::Device>& surface_velocities,
    const TensorView3D<Location::Device>& verts_wing,
    const TensorView2D<Location::Device>& areas,
    const TensorView2D<Location::Device>& mu,
    const linalg::float3& freestream
)
{
    for (i64 j = 0; j < areas.shape(1); j++) {
        for (i64 i = 0; i < areas.shape(0); i++) {
            const linalg::float3 v0{
                verts_wing(i, j, 0),
                verts_wing(i, j, 1),
                verts_wing(i, j, 2)
            }; // upper left
            const linalg::float3 v1{
                verts_wing(i+1, j, 0),
                verts_wing(i+1, j, 1),
                verts_wing(i+1, j, 2)
            }; // upper right
            const linalg::float3 v2{
                verts_wing(i+1, j+1, 0),
                verts_wing(i+1, j+1, 1),
                verts_wing(i+1, j+1, 2)
            }; // lower right
            const linalg::float3 v3{
                verts_wing(i, j+1, 0),
                verts_wing(i, j+1, 1),
                verts_wing(i, j+1, 2)
            }; // lower left
            const auto g_transform_l = panel_local_frame(v0, v1, v2, v3);
            const linalg::float4 freestream_local = linalg::mul(g_transform_l, linalg::float4{freestream.x, freestream.y, freestream.z, 0.0f});
            const linalg::float4 v0_local = linalg::mul(g_transform_l, linalg::float4{v0.x, v0.y, v0.z, 1.0f});
            const linalg::float4 v1_local = linalg::mul(g_transform_l, linalg::float4{v1.x, v1.y, v1.z, 1.0f});
            const linalg::float4 v2_local = linalg::mul(g_transform_l, linalg::float4{v2.x, v2.y, v2.z, 1.0f});
            const linalg::float4 v3_local = linalg::mul(g_transform_l, linalg::float4{v3.x, v3.y, v3.z, 1.0f});
            
            const linalg::float4 e0 = v1_local - v0_local;
            const linalg::float4 e1 = v2_local - v1_local;
            const linalg::float4 e2 = v3_local - v2_local;
            const linalg::float4 e3 = v0_local - v3_local;

            // const i64 ip1 = clamp(i+1, 0ll, areas.shape(0)-1);
            // const i64 im1 = clamp(i-1, 0ll, areas.shape(0)-1);
            const i64 ip1 = (i+1) % areas.shape(0);
            const i64 im1 = (i-1);
            const i64 jp1 = clamp(j+1, 0ll, areas.shape(1)-1);
            const i64 jm1 = clamp(j-1, 0ll, areas.shape(1)-1);
            
            // Gradient evaluation with Green-Gauss theorem
            linalg::float2 grad{0.0f, 0.0f};
            grad += 0.5f * (mu(i, jm1) + mu(i, j)) * rot_vec_cc_90(linalg::float2{e0.x, e0.y});
            grad += 0.5f * (mu(ip1, j) + mu(i, j)) * rot_vec_cc_90(linalg::float2{e1.x, e1.y});
            grad += 0.5f * (mu(i, jp1) + mu(i, j)) * rot_vec_cc_90(linalg::float2{e2.x, e2.y});
            grad += 0.5f * (mu(im1, j) + mu(i, j)) * rot_vec_cc_90(linalg::float2{e3.x, e3.y});
            grad /= areas(i, j);
            surface_velocities(i, j, 0) = - grad.x + freestream_local.x;
            surface_velocities(i, j, 1) = - grad.y + freestream_local.y;
        }
    }
}

void PM::pressure_coefficient(
    TensorView3D<Location::Device>& surface_velocities,
    TensorView2D<Location::Device>& cp
)
{
    const f32 u_ref = 1.0f;
    for (i64 j = 0; j < cp.shape(1); j++) {
        for (i64 i = 0; i < cp.shape(0); i++) {
            const linalg::float2 u_panel{surface_velocities(i, j, 0), surface_velocities(i, j, 1)};
            cp(i, j) = 1 - linalg::length2(u_panel) / (u_ref*u_ref);
        }
    }
}

void PM::run() {
    for (const auto& [vwing_init, vwing] : zip(verts_wing_init.views(), verts_wing.views())) {
        vwing_init.to(vwing);
    }
    const linalg::float3 freestream{1.0f, 0.0f, 0.0f};
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc.views(), normals.views(), areas.views());
    // flip normals (todo: automate this)
    for (i64 j = 0; j < normals.views()[0].shape(1); j++) {
        for (i64 i = 0; i < normals.views()[0].shape(0); i++) {
            normals.views()[0](i, j, 0) *= -1.0f;
            normals.views()[0](i, j, 1) *= -1.0f;
            normals.views()[0](i, j, 2) *= -1.0f;
        }
    }
    lhs.view().fill(0.f);
    rhs.view().fill(0.f);
    lhs_assemble(lhs.view(), colloc.views(), verts_wing.views());
    rhs_assemble(rhs.view(), colloc.views(), normals.views(), verts_wing.views(), freestream);
    mu_solver->factorize(lhs.view());
    mu_solver->solve(lhs.view(), rhs.view());
    rhs.view().to(mu.views()[0].reshape(mu.views()[0].size()));
    mu.views()[0].to(mu_prev.views()[0]); // steady state solution (dmu/dt = 0)
    surface_velocities(surf_vel.views()[0], verts_wing.views()[0], areas.views()[0], mu.views()[0], freestream);
    pressure_coefficient(surf_vel.views()[0], cp.views()[0]);
    
    // Note: for GPU backend need to transfer to device memory
    {
        const i64 j = 2;
        std::ofstream file_handle("pm_cylinder.txt");
        auto& colloc_m = colloc.views()[0];
        auto& cp_m = cp.views()[0];

        for (i64 i = 0; i < cp_m.shape(0); i++) {
            f32 theta = std::atan2(colloc_m(i, j, 1), colloc_m(i, j, 0));
            file_handle << theta << " " << cp_m(i, j) << "\n";
            std::printf("theta: %.3f | cp: %.3f\n", theta, cp_m(i, j));
        }
    }
}

int main(int  /*argc*/, char** /*argv*/) {
    const std::vector<std::string> meshes = {"../../../../mesh/cylinder.x"};
    const std::vector<std::string> backends = {"cpu"};

    auto simulations = tiny::make_combination(meshes, backends);

    for (const auto& [mesh_name, backend_name] : simulations) {
        PM simulation{backend_name, mesh_name};
        simulation.run();
    }
    return 0;
}

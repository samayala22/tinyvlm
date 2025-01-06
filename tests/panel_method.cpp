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

        Tensor2D<Location::Device> lhs{backend->memory.get()}; // (ns*nc)^2
        Tensor1D<Location::Device> rhs{backend->memory.get()}; // ns*nc
        
        std::unique_ptr<LU> mu_solver;

        void run();
    private:
        void alloc_buffers();
        void lhs_assemble(
            TensorView2D<Location::Device>& lhs,
            const MultiTensorView3D<Location::Device>& colloc,
            const MultiTensorView3D<Location::Device>& normals,
            const MultiTensorView3D<Location::Device>& verts_wing
        );
        void rhs_assemble(
            TensorView1D<Location::Device>& rhs,
            const MultiTensorView3D<Location::Device>& colloc,
            const MultiTensorView3D<Location::Device>& normals,
            const MultiTensorView3D<Location::Device>& verts_wing
        );
};

PM::PM(const std::string& backend_name, const std::string& mesh) : Simulation(backend_name, {mesh}) {
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

    lhs.init({n, n});
    rhs.init({n});

    mu_solver->init(lhs.view());
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
    if (F != G) {
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

// lhs is column major matrix, so we fill column by column
void PM::lhs_assemble(
    TensorView2D<Location::Device>& lhs,
    const MultiTensorView3D<Location::Device>& colloc,
    const MultiTensorView3D<Location::Device>& normals,
    const MultiTensorView3D<Location::Device>& verts_wing
) {
    auto& colloc_m = colloc[0];
    auto& normals_m = normals[0];
    auto& verts_wing_m = verts_wing[0];
    for (i64 j = 0; j < colloc_m.shape(1); j++) { // influencer
        for (i64 i = 0; i < colloc_m.shape(0); i++) { // influencer
            const linalg::float3 v0{verts_wing_m(i, j, 0), verts_wing_m(i, j, 1), verts_wing_m(i, j, 2)}; // upper left
            const linalg::float3 v1{verts_wing_m(i+1, j, 0), verts_wing_m(i+1, j, 1), verts_wing_m(i+1, j, 2)}; // upper right
            const linalg::float3 v2{verts_wing_m(i+1, j+1, 0), verts_wing_m(i+1, j+1, 1), verts_wing_m(i+1, j+1, 2)}; // lower right
            const linalg::float3 v3{verts_wing_m(i, j+1, 0), verts_wing_m(i, j+1, 1), verts_wing_m(i, j+1, 2)}; // lower left
            const i64 influencer_lidx = j * colloc_m.shape(0) + i;

            for (i64 jj = 0; jj < colloc_m.shape(1); jj++) { // influenced
                for (i64 ii = 0; ii < colloc_m.shape(0); ii++) { // influenced
                    const linalg::float3 colloc_inf{colloc_m(ii, jj, 0), colloc_m(ii, jj, 1), colloc_m(ii, jj, 2)};
                    const i64 influenced_lidx = jj * colloc_m.shape(0) + ii;

                    f32 influence = 0.0f;
                    influence += doublet_edge_influence(v0, v1, colloc_inf);
                    influence += doublet_edge_influence(v1, v2, colloc_inf);
                    influence += doublet_edge_influence(v2, v3, colloc_inf);
                    influence += doublet_edge_influence(v3, v0, colloc_inf);

                    lhs(influenced_lidx, influencer_lidx) = influence / (4.f * PI_f);
                }
            }
        }
    }
}

// Performs the matrix-free assmembly of the rhs with the source influence
void PM::rhs_assemble(
    TensorView1D<Location::Device>& rhs,
    const MultiTensorView3D<Location::Device>& colloc,
    const MultiTensorView3D<Location::Device>& normals,
    const MultiTensorView3D<Location::Device>& verts_wing
)
{
    auto& colloc_m = colloc[0];
    auto& normals_m = normals[0];
    auto& verts_wing_m = verts_wing[0];
    const linalg::float3 freestream{1.0f, 0.0f, 0.0f}; // flow freestream

    for (i64 j = 0; j < colloc_m.shape(1); j++) { // row
        for (i64 i = 0; i < colloc_m.shape(0); i++) { // row
            const i64 row_idx = j * colloc_m.shape(0) + i;
            const linalg::float3 colloc_inf{colloc_m(i, j, 0), colloc_m(i, j, 1), colloc_m(i, j, 2)};

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
                    const f32 sigma = linalg::dot(normal, freestream);

                    f32 influence = 0.0f;
                    influence += source_edge_influence(v0, v1, colloc_inf);
                    influence += source_edge_influence(v1, v2, colloc_inf);
                    influence += source_edge_influence(v2, v3, colloc_inf);
                    influence += source_edge_influence(v3, v0, colloc_inf);
                    influence *= - 1.0f / (4.f * PI_f); // can move this to the accumulated influence

                    accumulated_influence += sigma * influence;
                }
            }
            rhs(row_idx) = - accumulated_influence;
        }
    }
}

void PM::run() {
    backend->mesh_metrics(0.0f, verts_wing.views(), colloc.views(), normals.views(), areas.views());
    lhs.view().fill(0.f);
    rhs.view().fill(0.f);
    lhs_assemble(lhs.view(), colloc.views(), normals.views(), verts_wing.views());
    rhs_assemble(rhs.view(), colloc.views(), normals.views(), verts_wing.views());
    mu_solver->factorize(lhs.view());
    mu_solver->solve(lhs.view(), rhs.view());
    rhs.view().to(mu.views()[0].reshape(mu.views()[0].size()));
    mu.views()[0].to(mu_prev.views()[0]); // steady state solution (dmu/dt = 0)
}

int main(int  /*argc*/, char** /*argv*/) {
    const std::vector<std::string> meshes = {"../../../../mesh/cylinder.x"};
    const std::vector<std::string> backends = {"cpu"};

    auto simulations = tiny::make_combination(meshes, backends);

    for (const auto& [mesh_name, backend_name] : simulations) {
        // PM simulation{backend_name, mesh_name};
        // simulation.run();
    }
    return 0;
}

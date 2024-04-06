#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <functional> // std::function

#include "tinycombination.hpp"

#include "vlm.hpp"
#include "vlm_data.hpp"
#include "vlm_utils.hpp"

#define DEBUG_DISPLACEMENT_DATA

using namespace vlm;
using namespace linalg::ostream_overloads;

using tmatrix = linalg::alias::float4x4; // transformation matrix

class Kinematics {
    public:
    std::vector<std::function<tmatrix(f32 t)>> joints;
    Kinematics() = default;
    ~Kinematics() = default;

    void add(std::function<tmatrix(f32 t)>&& joint) {
        joints.push_back(std::move(joint));
    }

    tmatrix displacement(f32 t) {
        tmatrix result = linalg::identity;
        for (const auto& joint : joints) {
            result = linalg::mul(result, joint(t));
        }
        return result;
    }

    tmatrix relative_displacement(f32 t0, f32 t1) {
        return linalg::mul(displacement(t1), linalg::inverse(displacement(t0)));
    }

    linalg::alias::float4 velocity(f32 t, const linalg::alias::float4& vertex) {
        return (linalg::mul(relative_displacement(t, t+EPS_sqrt_f), vertex)-vertex)/EPS_sqrt_f;
    }

    f32 velocity_magnitude(f32 t, const linalg::alias::float4& vertex) {
        return linalg::length(velocity(t, vertex));
    }
};

template<typename T>
void dump_buffer(std::ofstream& stream, T* start, T* end) {
    for (T* it = start; it != end; it++) {
        stream << *it << " ";
    }
    stream << "\n";
}

int main() {
    // const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_2x8.x"};
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_5x200.x"};

    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);


    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 t_final = 30.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 0.5f; // reduced frequency

    const f32 omega = k * u_inf / (2*b);

    Kinematics kinematics{};

    // Define wing motion
    kinematics.add([=](f32 t) {
        return linalg::translation_matrix(linalg::alias::float3{
            0.0f,
            0.0f,
            amplitude * std::sin(omega* t)
        });
    });

    kinematics.add([=](f32 t) {
        return linalg::translation_matrix(linalg::alias::float3{
            -u_inf*t,
            0.0f,
            0.0f
        });
    });

    for (const auto& [mesh_name, backend_name] : solvers) {
        const std::unique_ptr<Mesh> mesh = create_mesh(mesh_name);

        // Pre-calculate timesteps to determine wake size
        std::vector<f32> vec_t; // timesteps
        vec_t.push_back(0.0f);
        for (f32 t = 0.0f; t < t_final;) {
            // TODO: this is currently not accurate for rotational motion
            const f32 dt = mesh->panel_length(mesh->nc-1, 0) / kinematics.velocity_magnitude(t, {0.f, 0.f, 0.f, 1.f});
            t += dt;
            vec_t.push_back(t);
        }

        #ifdef DEBUG_DISPLACEMENT_DATA
        std::ofstream wing_data("wing_data.txt");
        std::ofstream wake_data("wake_data.txt");
        std::ofstream cl_data("cl_data.txt");
        
        wing_data << mesh->nc << " " << mesh->ns << "\n";
        wing_data << vec_t.size() - 1 << "\n\n";

        wake_data << mesh->ns << "\n";
        wake_data << vec_t.size() - 1 << "\n\n";
        #endif

        mesh->resize_wake(vec_t.size()-1); // +1 for the initial pose
        const std::unique_ptr<Backend> backend = create_backend(backend_name, *mesh); // create after mesh has been resized

        // Precompute the LHS since wing geometry is constant
        const FlowData flow_dummy{0.0f, 0.0f, 1.0f, 1.0f};
        backend->compute_lhs(flow_dummy);
        backend->lu_factor();

        // Unsteady loop
        std::cout << "Timesteps: " << vec_t.size() << "\n";
        for (u64 i = 0; i < vec_t.size()-1; i++) {
            assert(mesh->current_nw == i); // dumb assert

            #ifdef DEBUG_DISPLACEMENT_DATA
            dump_buffer(wing_data, mesh->v.x.data(), mesh->v.x.data() + mesh->nb_vertices_wing());
            dump_buffer(wing_data, mesh->v.y.data(), mesh->v.y.data() + mesh->nb_vertices_wing());
            dump_buffer(wing_data, mesh->v.z.data(), mesh->v.z.data() + mesh->nb_vertices_wing());
            wing_data << "\n";

            const u64 wake_start = (mesh->nc + mesh->nw - i) * (mesh->ns + 1);
            const u64 wake_end = mesh->nb_vertices_total();
            std::cout << "Buffer size: " << mesh->v.x.size() << " | " << wake_start << " | " << wake_end << std::endl;

            dump_buffer(wake_data, mesh->v.x.data() + wake_start, mesh->v.x.data() + wake_end);
            dump_buffer(wake_data, mesh->v.y.data() + wake_start, mesh->v.y.data() + wake_end);
            dump_buffer(wake_data, mesh->v.z.data() + wake_start, mesh->v.z.data() + wake_end);
            wake_data << "\n";
            #endif

            const f32 t = vec_t[i];
            const f32 dt = vec_t[i+1] - t;
            auto base_vertex = mesh->get_v0(0);
            auto base_velocity = kinematics.velocity(vec_t[i], {base_vertex[0], base_vertex[1], base_vertex[2], 1.0f});
            const FlowData flow{linalg::alias::float3{base_velocity[0], base_velocity[1], base_velocity[2]}, 1.0f};
            backend->compute_rhs(flow);
            backend->add_wake_influence(flow);
            backend->lu_solve();
            backend->compute_delta_gamma();
            if (i > 0) {
                // TODO: this should take a vector of local velocities magnitude because it can be different for each point on the mesh
                const f32 cl_unsteady = backend->compute_coefficient_unsteady_cl(flow, dt, mesh->s_ref, 0, mesh->ns);
                std::printf("t: %f, CL: %f\n", t, cl_unsteady);
                #ifdef DEBUG_DISPLACEMENT_DATA
                cl_data << t << " " << cl_unsteady << "\n";
                #endif
            }
            mesh->move(kinematics.relative_displacement(t, t+dt));
            backend->shed_gamma();
        }
    }
 
    return 0;
}
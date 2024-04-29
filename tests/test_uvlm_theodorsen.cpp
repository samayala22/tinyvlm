#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <functional> // std::function

#include "tinycombination.hpp"

#include "tinytimer.hpp"
#include "vlm.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_executor.hpp"

#define DEBUG_DISPLACEMENT_DATA

using namespace vlm;
using namespace linalg::ostream_overloads;

using tmatrix = linalg::alias::float4x4; // transformation matri

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
        //std::printf("t0: %.10f, t1: %.10f ", t0, t1);
        return linalg::mul(displacement(t1), linalg::inverse(displacement(t0)));
    }

    linalg::alias::float4 velocity(f32 t, const linalg::alias::float4& vertex) {
        //const f32 EPS = std::max(10.f * t *EPS_f, EPS_f); // adaptive epsilon
        const f32 EPS = std::max(std::sqrt(t) * EPS_sqrt_f, EPS_f);
        return (linalg::mul(relative_displacement(t, t+EPS), vertex)-vertex)/EPS;
        //return (linalg::mul(relative_displacement(t, t+EPS), vertex) - linalg::mul(relative_displacement(t, t-EPS), vertex))/ (2*EPS); // central diff
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

template<typename T>
void print_buffer(const T* start, u64 size) {
    std::cout << "[";
    for (u64 i = 0; i < size; i++) {
        std::cout << start[i] << ",";
    }
    std::cout << "]\n";
}

int main() {
    const tiny::ScopedTimer timer("UVLM TOTAL");
    // vlm::Executor::instance(1);
    // const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_2x8.x"};
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_10x5.x"};

    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 4.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 0.6f; // reduced frequency
    const f32 omega = k * 2.0f * u_inf / (2*b);
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods

    Kinematics kinematics{};

    // Periodic heaving
    // kinematics.add([=](f32 t) {
    //     return linalg::translation_matrix(linalg::alias::float3{
    //         -u_inf*t, // freestream
    //         0.0f,
    //         amplitude * std::sin(omega * t) // heaving
    //     });
    // });

    // Periodic pitching
    kinematics.add([=](f32 t) {
        return linalg::translation_matrix(linalg::alias::float3{
            -u_inf*t, // freestream
            0.0f,
            0.0f
        });
    });
    kinematics.add([=](f32 t) {
        return linalg::rotation_matrix(
            linalg::alias::float3{0.0f, 0.0f, 0.0f},
            linalg::alias::float3{0.0f, 1.0f, 0.0f},
            to_radians(std::sin(omega * t))
        );
    });
    
    // Sudden acceleration
    // const f32 alpha = to_radians(5.0f);
    // kinematics.add([=](f32 t) {
    //     return linalg::translation_matrix(linalg::alias::float3{
    //         -u_inf*cos(alpha)*t,
    //         0.0f,
    //         -u_inf*sin(alpha)*t
    //     });
    // });

    for (const auto& [mesh_name, backend_name] : solvers) {
        const std::unique_ptr<Mesh> mesh = create_mesh(mesh_name);

        // Pre-calculate timesteps to determine wake size
        // Note: calculation should be made on the four corners of the wing
        std::vector<f32> vec_t; // timesteps
        SoA_3D_t<f32> velocities;
        velocities.resize(mesh->nb_panels_wing());
        {
            SoA_3D_t<f32> trailing_vertices;
            trailing_vertices.resize(mesh->ns+1);
            std::copy(mesh->v.x.data() + mesh->nb_vertices_wing() - mesh->ns - 1, mesh->v.x.data() + mesh->nb_vertices_wing(), trailing_vertices.x.data());
            std::copy(mesh->v.y.data() + mesh->nb_vertices_wing() - mesh->ns - 1, mesh->v.y.data() + mesh->nb_vertices_wing(), trailing_vertices.y.data());
            std::copy(mesh->v.z.data() + mesh->nb_vertices_wing() - mesh->ns - 1, mesh->v.z.data() + mesh->nb_vertices_wing(), trailing_vertices.z.data());
            const f32 segment_chord = mesh->panel_length(mesh->nc-1, 0); // TODO: this can be variable
            
            vec_t.push_back(0.0f);
            for (f32 t = 0.0f; t < t_final;) {
                f32 dt = segment_chord / kinematics.velocity_magnitude(t, {trailing_vertices.x[0], trailing_vertices.y[0], trailing_vertices.z[0], 1.0f});
                for (u64 i = 1; i < trailing_vertices.size; i++) {
                    dt = std::min(dt, segment_chord / kinematics.velocity_magnitude(t, {trailing_vertices.x[i], trailing_vertices.y[i], trailing_vertices.z[i], 1.0f}));
                }

                auto transform = kinematics.relative_displacement(t, t+dt);
                for (u64 i = 0; i < trailing_vertices.size; i++) {
                    const linalg::alias::float4 transformed_pt = linalg::mul(transform, linalg::alias::float4{trailing_vertices.x[i], trailing_vertices.y[i], trailing_vertices.z[i], 1.f});
                    trailing_vertices.x[i] = transformed_pt.x;
                    trailing_vertices.y[i] = transformed_pt.y;
                    trailing_vertices.z[i] = transformed_pt.z;
                }
                t += dt;
                vec_t.push_back(t);
            }
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
        backend->compute_lhs();
        backend->lu_factor();

        // Unsteady loop
        std::cout << "SIMULATION NB OF TIMESTEPS: " << vec_t.size() << "\n";
        for (u64 i = 0; i < vec_t.size()-1; i++) {
            #ifdef DEBUG_DISPLACEMENT_DATA
            dump_buffer(wing_data, mesh->v.x.data(), mesh->v.x.data() + mesh->nb_vertices_wing());
            dump_buffer(wing_data, mesh->v.y.data(), mesh->v.y.data() + mesh->nb_vertices_wing());
            dump_buffer(wing_data, mesh->v.z.data(), mesh->v.z.data() + mesh->nb_vertices_wing());
            wing_data << "\n";

            const u64 wake_start = (mesh->nc + mesh->nw - i) * (mesh->ns + 1);
            const u64 wake_end = mesh->nb_vertices_total();
            // std::cout << "Buffer size: " << mesh->v.x.size() << " | " << wake_start << " | " << wake_end << std::endl;
 
            dump_buffer(wake_data, mesh->v.x.data() + wake_start, mesh->v.x.data() + wake_end);
            dump_buffer(wake_data, mesh->v.y.data() + wake_start, mesh->v.y.data() + wake_end);
            dump_buffer(wake_data, mesh->v.z.data() + wake_start, mesh->v.z.data() + wake_end);
            wake_data << "\n";
            #endif

            const f32 t = vec_t[i];
            const f32 dt = vec_t[i+1] - t;
            std::cout << "\n----------------\n" << "T = " << t << "\n";

            for (u64 idx = 0; idx < mesh->nb_panels_wing(); idx++) {
                const linalg::alias::float4 colloc_pt{mesh->colloc.x[idx], mesh->colloc.y[idx], mesh->colloc.z[idx], 1.0f};
                auto local_velocity = -kinematics.velocity(t, colloc_pt);
                velocities.x[idx] = local_velocity[0];
                velocities.y[idx] = local_velocity[1];
                velocities.z[idx] = local_velocity[2];
            }

            backend->compute_rhs(velocities);
            backend->add_wake_influence();
            backend->lu_solve();
            backend->compute_delta_gamma();
            if (i > 0) {
                // TODO: this should take a vector of local velocities magnitude because it can be different for each point on the mesh
                const f32 cl_unsteady = backend->compute_coefficient_unsteady_cl(velocities, dt, mesh->s_ref, 0, mesh->ns);

                std::printf("t: %f, CL: %f\n", t, cl_unsteady);
                #ifdef DEBUG_DISPLACEMENT_DATA
                cl_data << t << " " << mesh->v.z[0] << " " << cl_unsteady << " " << std::sin(omega * t) << "\n";
                #endif
            }
            backend->shed_gamma(); // shed before moving & incrementing currentnw
            mesh->move(kinematics.relative_displacement(t, t+dt));
        }
    }

    return 0;
}
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

// TODO: add caching for transformation matrices 
class Kinematics {
    public:
    Kinematics() = default;
    ~Kinematics() = default;

    void add(std::function<tmatrix(f32 t)>&& joint) {
        m_joints.push_back(std::move(joint));
    }

    tmatrix displacement(f32 t, u64 n = 0) {
        tmatrix result = linalg::identity;
        const u64 end_joint = n == 0 ? m_joints.size() : n;
        for (u64 i = 0; i < end_joint; i++) {
            result = linalg::mul(result, m_joints[i](t));
        }
        return result;
    }

    tmatrix relative_displacement(f32 t0, f32 t1, u64 n = 0) {
        return linalg::mul(displacement(t1, n), linalg::inverse(displacement(t0, n)));
    }

    // Compute the instantaneous velocity vector at a given point at a given time for n joints (starting from the first, 0 = all joints)
    linalg::alias::float4 velocity(f32 t, const linalg::alias::float4& vertex, u64 n = 0) {
        const f32 EPS = EPS_sqrt_f;
        //return (linalg::mul(relative_displacement(t, t+EPS), vertex)-vertex)/EPS;
        return (linalg::mul(relative_displacement(t, t+EPS, n), vertex) - linalg::mul(relative_displacement(t, t-EPS, n), vertex))/ (2*EPS); // central diff
    }

    f32 velocity_magnitude(f32 t, const linalg::alias::float4& vertex) {
        return linalg::length(velocity(t, vertex));
    }

    private:
    std::vector<std::function<tmatrix(f32 t)>> m_joints;
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

    const u64 ni = 20;
    const u64 nj = 5;
    // vlm::Executor::instance(1);
    //const std::vector<std::string> meshes = {"../../../../mesh/rectangular_5x10.x"};
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_" + std::to_string(ni) + "x" + std::to_string(nj) + ".x"};
    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 3.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 0.7; // reduced frequency
    const f32 omega = k * 2.0f * u_inf / (2*b);
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods
    //const f32 t_final = 5.0f;

    Kinematics kinematics{};

    const f32 initial_angle = 0.0f;

    const tmatrix initial_pose = linalg::rotation_matrix(
        linalg::alias::float3{0.0f, 0.0f, 0.0f}, // take into account quarter chord panel offset
        linalg::alias::float3{0.0f, 1.0f, 0.0f},
        to_radians(initial_angle)
    );

    // Periodic heaving
    kinematics.add([=](f32 t) {
        return linalg::translation_matrix(linalg::alias::float3{
            -u_inf*t,
            0.0f,
            0.0f
        });
    });
    kinematics.add([=](f32 t) {
        return linalg::translation_matrix(linalg::alias::float3{
            0.0f,
            0.0f,
            amplitude * std::sin(omega * t) // heaving
        });
    });

    // Periodic pitching
    // kinematics.add([=](f32 t) {
    //     return linalg::translation_matrix(linalg::alias::float3{
    //         -u_inf*t, // freestream
    //         0.0f,
    //         0.0f
    //     });
    // });
    // kinematics.add([=](f32 t) {
    //     return linalg::rotation_matrix(
    //         linalg::alias::float3{0.25f, 0.0f, 0.0f},
    //         linalg::alias::float3{0.0f, 1.0f, 0.0f},
    //         to_radians(std::sin(omega * t))
    //     );
    // });
    
    // Sudden acceleration
    // const f32 alpha = to_radians(5.0f);
    // kinematics.add([=](f32 t) {
    //     return linalg::translation_matrix(linalg::alias::float3{
    //         -u_inf*cos(alpha)*t,
    //         0.0f,
    //         -u_inf*sin(alpha)*t
    //     });
    // });
    // kinematics.add([=](f32 t) {
    //     return linalg::translation_matrix(linalg::alias::float3{
    //         -u_inf*t,
    //         0.0f,
    //         0.0f
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
            
            std::cout << "Timestep calculation\n";
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

        cl_data << k << "\n";

        wing_data << mesh->nc << " " << mesh->ns << "\n";
        wing_data << vec_t.size() - 1 << "\n\n";

        wake_data << mesh->ns << "\n";
        wake_data << vec_t.size() - 1 << "\n\n";
        #endif

        mesh->resize_wake(vec_t.size()-1); // +1 for the initial pose
        const std::unique_ptr<Backend> backend = create_backend(backend_name, *mesh); // create after mesh has been resized
        
        // Initial position
        for (u64 i = 0; i < mesh->nb_vertices_wing(); i++) {
            const linalg::alias::float4 transformed_pt = linalg::mul(initial_pose, linalg::alias::float4{mesh->v.x[i], mesh->v.y[i], mesh->v.z[i], 1.f});
            mesh->v.x[i] = transformed_pt.x;
            mesh->v.y[i] = transformed_pt.y;
            mesh->v.z[i] = transformed_pt.z;
        }
        mesh->compute_metrics_wing();

        // Precompute the LHS since wing geometry is constant
        backend->compute_lhs();
        backend->lu_factor();

        // Unsteady loop
        std::cout << "SIMULATION NB OF TIMESTEPS: " << vec_t.size() << "\n";
        f32 avg_vel_error = 0.0f;
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

            linalg::alias::float3 freestream;
            for (u64 idx = 0; idx < mesh->nb_panels_wing(); idx++) {
                const linalg::alias::float4 colloc_pt{mesh->colloc.x[idx], mesh->colloc.y[idx], mesh->colloc.z[idx], 1.0f};
                // auto local_velocity = -kinematics.velocity(t, colloc_pt);
                // velocities.x[idx] = local_velocity.x;
                // velocities.y[idx] = local_velocity.y;
                // velocities.z[idx] = local_velocity.z;

                velocities.x[idx] = u_inf;
                velocities.y[idx] = 0.0f;
                velocities.z[idx] = - omega * amplitude * std::cos(omega * t);

                if (idx == 0) {
                    linalg::alias::float4 freestream_vel = -kinematics.velocity(t, colloc_pt, 1);
                    freestream = {freestream_vel.x, freestream_vel.y, freestream_vel.z};
                    // std::cout << "Freestream: " << freestream << "\n";
                    // const f32 analytical_vel = - amplitude * omega * std::cos(omega * t);
                    // const f32 rel_error = 100.0f * std::abs((analytical_vel - local_velocity.z) / analytical_vel);
                    // std::cout << "vel error:" << rel_error << "%\n";
                    // avg_vel_error += rel_error;
                }
            }

            backend->compute_rhs(velocities);
            backend->add_wake_influence();
            backend->lu_solve();
            backend->compute_delta_gamma();
            if (i > 0) {
                // TODO: this should take a vector of local velocities magnitude because it can be different for each point on the mesh
                const f32 cl_unsteady = backend->compute_coefficient_unsteady_cl(freestream, velocities, dt, mesh->s_ref, 0, mesh->ns);

                std::printf("t: %f, CL: %f\n", t, cl_unsteady);
                #ifdef DEBUG_DISPLACEMENT_DATA
                cl_data << t << " " << mesh->v.z[0] << " " << cl_unsteady << " " << std::sin(omega * t) << "\n";
                #endif
            }
            //backend->wake_rollup(dt);
            backend->shed_gamma(); // shed before moving & incrementing currentnw
            mesh->move(kinematics.relative_displacement(t, t+dt));
            mesh->frame = linalg::mul(mesh->frame, kinematics.relative_displacement(t, t+dt));
        }

        avg_vel_error /= (f32)(vec_t.size()-1);
        std::cout << "Average velocity error: " << avg_vel_error << "%\n";
    }

    return 0;
}
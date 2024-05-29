#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <functional> // std::function

#include "tinycombination.hpp"
#include "tinyad.hpp"
#include "tinytimer.hpp"

#include "vlm.hpp"
#include "vlm_data.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"
#include "vlm_executor.hpp"

// #define DEBUG_DISPLACEMENT_DATA

using namespace vlm;
using namespace linalg::ostream_overloads;

using tmatrix = linalg::mat<fwd::Float,4,4>;

linalg::alias::float4x4 dual_to_float(const tmatrix& m) {
    return {
        {m.x.x.val(), m.x.y.val(), m.x.z.val(), m.x.w.val()},
        {m.y.x.val(), m.y.y.val(), m.y.z.val(), m.y.w.val()},
        {m.z.x.val(), m.z.y.val(), m.z.z.val(), m.z.w.val()},
        {m.w.x.val(), m.w.y.val(), m.w.z.val(), m.w.w.val()}
    };
}

class Kinematics {
    public:
    Kinematics() = default;
    ~Kinematics() = default;

    void add(std::function<tmatrix(const fwd::Float& t)>&& joint) {
        m_joints.push_back(std::move(joint));
    }

    tmatrix displacement(float t, u64 n) {
        fwd::Float t_dual{t, 1.f};
        tmatrix result = linalg::identity;
        for (u64 i = 0; i < n; i++) {
            result = linalg::mul(result, m_joints[i](t_dual));
        }
        return result;
    }
    tmatrix displacement(float t) {return displacement(t, m_joints.size());}

    linalg::alias::float3 velocity(const tmatrix& transform, const linalg::vec<fwd::Float,4> vertex) {
        linalg::vec<fwd::Float,4> new_pt = linalg::mul(transform, vertex);
        return {new_pt.x.grad(), new_pt.y.grad(), new_pt.z.grad()};
    }

    f32 velocity_magnitude(const tmatrix& transform, const linalg::vec<fwd::Float,4> vertex) {
        return linalg::length(velocity(transform, vertex));
    }

    private:
    std::vector<std::function<tmatrix(const fwd::Float& t)>> m_joints;
};

template<class T> linalg::mat<T,4,4> translation_matrix(const linalg::vec<T,3> & translation) { return {{1,0,0,0},{0,1,0,0},{0,0,1,0},{translation,1}}; }

template<class T> 
linalg::mat<T,3,3> skew_matrix (const linalg::vec<T,3> & a) {
 return {{0, a.z, -a.y}, {-a.z, 0, a.x}, {a.y, -a.x, 0}}; 
}

template<class T> linalg::mat<T,4,4> rotation_matrix   (const linalg::vec<T,3> & point, const linalg::vec<T,3> & axis, T angle) {
    using std::sin; using std::cos;
    using fwd::sin; using fwd::cos;
    const linalg::mat<T,3,3> skew_mat = skew_matrix<T>(axis);
    const linalg::mat<T,3,3> i = linalg::identity;

    const linalg::mat<T,3,3> rodrigues = i + sin(angle)*skew_mat + (1.f-cos(angle))*linalg::mul(skew_mat, skew_mat);
    const linalg::vec<T,3> trans_part = linalg::mul((i - rodrigues), point);
    return {
        {rodrigues.x.x, rodrigues.x.y, rodrigues.x.z, 0}, // 1st col
        {rodrigues.y.x, rodrigues.y.y, rodrigues.y.z, 0},
        {rodrigues.z.x, rodrigues.z.y, rodrigues.z.z, 0},
        {trans_part.x, trans_part.y, trans_part.z, 1}
    };
}

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

    const u64 ni = 56;
    const u64 nj = 5;
    // vlm::Executor::instance(1);
    //const std::vector<std::string> meshes = {"../../../../mesh/rectangular_5x10.x"};
    const std::vector<std::string> meshes = {"../../../../mesh/infinite_rectangular_" + std::to_string(ni) + "x" + std::to_string(nj) + ".x"};
    const std::vector<std::string> backends = get_available_backends();

    auto solvers = tiny::make_combination(meshes, backends);

    // Geometry
    const f32 b = 0.5f; // half chord

    // Define simulation length
    const f32 cycles = 4.0f;
    const f32 u_inf = 1.0f; // freestream velocity
    const f32 amplitude = 0.1f; // amplitude of the wing motion
    const f32 k = 3.0; // reduced frequency
    const f32 omega = k * 2.0f * u_inf / (2*b);
    const f32 t_final = cycles * 2.0f * PI_f / omega; // 4 periods
    //const f32 t_final = 5.0f;

    Kinematics kinematics{};

    const f32 initial_angle = 0.0f;

    const auto initial_pose = rotation_matrix(
        linalg::alias::float3{0.0f, 0.0f, 0.0f}, // take into account quarter chord panel offset
        linalg::alias::float3{0.0f, 1.0f, 0.0f},
        to_radians(initial_angle)
    );

    // Periodic heaving
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    // });
    // kinematics.add([=](const fwd::Float& t) {
    //     return translation_matrix<fwd::Float>({0.f, 0.f, amplitude * fwd::sin(omega * t)});
    // });

    // Periodic pitching
    kinematics.add([=](const fwd::Float& t) {
        return translation_matrix<fwd::Float>({-u_inf * t, 0.f, 0.f});
    });
    const f32 to_rad = PI_f / 180.0f;
    kinematics.add([=](const fwd::Float& t) {
        return rotation_matrix<fwd::Float>({0.25f, 0.0f, 0.0f},{0.0f, 1.0f, 0.0f}, to_rad * fwd::sin(omega * t));
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
    // kinematics.add([=](f32 t) {
    //     return linalg::translation_matrix(linalg::alias::float3{
    //         -u_inf*t,
    //         0.0f,
    //         0.0f
    //     });
    // });

    for (const auto& [mesh_name, backend_name] : solvers) {
        const std::unique_ptr<Mesh> mesh = create_mesh(mesh_name);

        // Initial position
        for (u64 i = 0; i < mesh->nb_vertices_wing(); i++) {
            const linalg::alias::float4 transformed_pt = linalg::mul(initial_pose, linalg::alias::float4{mesh->v.x[i], mesh->v.y[i], mesh->v.z[i], 1.f});
            mesh->v.x[i] = transformed_pt.x;
            mesh->v.y[i] = transformed_pt.y;
            mesh->v.z[i] = transformed_pt.z;
        }
        mesh->compute_metrics_wing(); // compute new metrics after initial position ajustement

        SoA_3D_t<f32> origin_pos;
        SoA_3D_t<f32> origin_colloc;
        origin_pos.resize(mesh->nb_vertices_wing());
        origin_colloc.resize(mesh->nb_panels_wing());
        std::copy(mesh->v.x.data(), mesh->v.x.data() + mesh->nb_vertices_wing(), origin_pos.x.data());
        std::copy(mesh->v.y.data(), mesh->v.y.data() + mesh->nb_vertices_wing(), origin_pos.y.data());
        std::copy(mesh->v.z.data(), mesh->v.z.data() + mesh->nb_vertices_wing(), origin_pos.z.data());
        std::copy(mesh->colloc.x.data(), mesh->colloc.x.data() + mesh->nb_panels_wing(), origin_colloc.x.data());
        std::copy(mesh->colloc.y.data(), mesh->colloc.y.data() + mesh->nb_panels_wing(), origin_colloc.y.data());
        std::copy(mesh->colloc.z.data(), mesh->colloc.z.data() + mesh->nb_panels_wing(), origin_colloc.z.data());

        // Pre-calculate timesteps to determine wake size
        // Note: calculation should be made on the four corners of the wing
        std::vector<f32> vec_t; // timesteps
        SoA_3D_t<f32> velocities;
        velocities.resize(mesh->nb_panels_wing());
        {
            SoA_3D_t<f32> trailing_vertices;
            trailing_vertices.resize(mesh->ns+1);
            const u64 trailing_begin = mesh->nc * (mesh->ns + 1);
            std::copy(mesh->v.x.data() + trailing_begin, mesh->v.x.data() + mesh->nb_vertices_wing(), trailing_vertices.x.data());
            std::copy(mesh->v.y.data() + trailing_begin, mesh->v.y.data() + mesh->nb_vertices_wing(), trailing_vertices.y.data());
            std::copy(mesh->v.z.data() + trailing_begin, mesh->v.z.data() + mesh->nb_vertices_wing(), trailing_vertices.z.data());
            const f32 segment_chord = mesh->panel_length(mesh->nc-1, 0); // TODO: this can be variable
            
            std::cout << "Timestep calculation\n";
            vec_t.push_back(0.0f);
            for (f32 t = 0.0f; t < t_final;) {
                const auto total_transform = kinematics.displacement(t);
                f32 dt = segment_chord / kinematics.velocity_magnitude(total_transform, {origin_pos.x[trailing_begin], origin_pos.y[trailing_begin], origin_pos.z[trailing_begin], 1.0f});
                for (u64 i = 1; i < trailing_vertices.size; i++) {
                    dt = std::min(dt, segment_chord / kinematics.velocity_magnitude(total_transform, {origin_pos.x[trailing_begin+i], origin_pos.y[trailing_begin+i], origin_pos.z[trailing_begin+i], 1.0f}));
                }
                t += dt;
                vec_t.push_back(t);
            }
        }

        std::ofstream cl_data("cl_data.txt");
        cl_data << k << "\n";

        #ifdef DEBUG_DISPLACEMENT_DATA
        std::ofstream wing_data("wing_data.txt");
        std::ofstream wake_data("wake_data.txt");

        wing_data << mesh->nc << " " << mesh->ns << "\n";
        wing_data << vec_t.size() - 1 << "\n\n";

        wake_data << mesh->ns << "\n";
        wake_data << vec_t.size() - 1 << "\n\n";
        #endif

        mesh->resize_wake(vec_t.size()-1); // dont account initial position
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
 
            dump_buffer(wake_data, mesh->v.x.data() + wake_start, mesh->v.x.data() + wake_end);
            dump_buffer(wake_data, mesh->v.y.data() + wake_start, mesh->v.y.data() + wake_end);
            dump_buffer(wake_data, mesh->v.z.data() + wake_start, mesh->v.z.data() + wake_end);
            wake_data << "\n";
            #endif

            const f32 t = vec_t[i];
            const f32 dt = vec_t[i+1] - t;
            const auto total_transform = kinematics.displacement(t);
            const auto freestream_transform = kinematics.displacement(t,1);

            linalg::alias::float3 freestream;
            for (u64 idx = 0; idx < mesh->nb_panels_wing(); idx++) {
                auto local_velocity = -kinematics.velocity(total_transform, {origin_colloc.x[idx], origin_colloc.y[idx], origin_colloc.z[idx], 1.0f});
                velocities.x[idx] = local_velocity.x;
                velocities.y[idx] = local_velocity.y;
                velocities.z[idx] = local_velocity.z;

                if (idx == 0) {
                    freestream = -kinematics.velocity(freestream_transform, {origin_colloc.x[idx], origin_colloc.y[idx], origin_colloc.z[idx], 1.0f});
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
                cl_data << t << " " << mesh->v.z[0] << " " << cl_unsteady << " " << std::sin(omega * t) << "\n";
            }
            // backend->wake_rollup(dt);
            backend->shed_gamma(); // shed before moving & incrementing currentnw
            const auto transform = dual_to_float(kinematics.displacement(t+dt));
            mesh->move(transform, origin_pos);
            const linalg::alias::float4x4 init_frame = linalg::identity;
            mesh->frame = linalg::mul(transform, init_frame);
        }
    }

    return 0;
}
#include "vlm_memory.hpp"
#include "tinyad.hpp"
#include "linalg.h"

#include <cmath>
#include <cstdio>
#include <cassert>
#include <string>

#define CHECK(condition)                     \
    do {                                            \
        if (!(condition)) {                         \
            std::fprintf(stderr,                         \
                    "Assertion failed: %s\n"        \
                    "File: %s, Line: %d\n",        \
                    #condition, __FILE__, __LINE__);\
            std::abort();                                \
        }                                           \
    } while (0)

bool APPROX(float a, float b) { return std::abs(a - b) < 1e-6; }

using namespace vlm;

using KinematicMatrixDual = linalg::mat<fwd::Float,4,4>;
using KinematicMatrix = linalg::mat<f32,4,4>;

inline linalg::float4x4 dual_to_float(const KinematicMatrixDual& m) {
    return {
        {m.x.x.val(), m.x.y.val(), m.x.z.val(), m.x.w.val()},
        {m.y.x.val(), m.y.y.val(), m.y.z.val(), m.y.w.val()},
        {m.z.x.val(), m.z.y.val(), m.z.z.val(), m.z.w.val()},
        {m.w.x.val(), m.w.y.val(), m.w.z.val(), m.w.w.val()}
    };
}

template<class T> linalg::mat<T,4,4> translation_matrix(const linalg::vec<T,3> & translation) { return {{1,0,0,0},{0,1,0,0},{0,0,1,0},{translation,1}}; }

template<class T> 
linalg::mat<T,3,3> skew_matrix (const linalg::vec<T,3> & a) {
 return {{0, a.z, -a.y}, {-a.z, 0, a.x}, {a.y, -a.x, 0}}; 
}

template<class T> linalg::mat<T,4,4> rotation_matrix(const linalg::vec<T,3> & point, const linalg::vec<T,3> & axis, T angle) {
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


class KinematicNode {
private:
    std::function<KinematicMatrixDual(const fwd::Float& t)> m_transform;
    KinematicNode* m_parent = nullptr;

public:
    KinematicNode(std::function<KinematicMatrixDual(const fwd::Float& t)> transform = [](const fwd::Float&) { return linalg::identity; }) 
        : m_transform(std::move(transform)) {}

    KinematicNode* after(KinematicNode* parent) {
        m_parent = parent;
        return this;
    }

    KinematicMatrixDual transform_dual(float t) const {
        fwd::Float t_dual{t, 1.f};
        KinematicMatrixDual result = m_transform(t_dual);
        if (m_parent) {
            return linalg::mul(m_parent->transform_dual(t), result);
        }
        return result;
    }

    KinematicMatrix transform(float t) const {
        return dual_to_float(transform_dual(t));
    }

    linalg::float3 linear_velocity(float t, const linalg::float3 vertex) const {
        linalg::vec<fwd::Float,4> new_pt = linalg::mul(transform_dual(t), {vertex.x, vertex.y, vertex.z, 1.0f});
        return {new_pt.x.grad(), new_pt.y.grad(), new_pt.z.grad()};
    }

    linalg::float3 angular_velocity(float t) const {
        // Step 1: Get the dual transform
        KinematicMatrixDual dual_transform = transform_dual(t);

        // Step 2: Extract R and R_dot from the dual transform
        // R(t) is the value part, R_dot(t) is the gradient part
        linalg::mat<f32,3,3> R = {
            {dual_transform.x.x.val(), dual_transform.x.y.val(), dual_transform.x.z.val()},
            {dual_transform.y.x.val(), dual_transform.y.y.val(), dual_transform.y.z.val()},
            {dual_transform.z.x.val(), dual_transform.z.y.val(), dual_transform.z.z.val()}
        };   

        linalg::mat<f32,3,3> R_dot = {
            {dual_transform.x.x.grad(), dual_transform.x.y.grad(), dual_transform.x.z.grad()},
            {dual_transform.y.x.grad(), dual_transform.y.y.grad(), dual_transform.y.z.grad()},
            {dual_transform.z.x.grad(), dual_transform.z.y.grad(), dual_transform.z.z.grad()}
        };

        // Step 3: Compute Omega = R_dot * R^T
        linalg::mat<f32,3,3> Omega = linalg::mul(R_dot, linalg::transpose(R));

        // Step 4: Extract angular velocity from Omega
        // Omega is skew-symmetric: Omega = [ [0, -wz, wy],
        //                                  [wz, 0, -wx],
        //                                  [-wy, wx, 0] ]

        return {Omega.y.z, Omega.z.x, Omega.x.y};
    }
};

class KinematicsTree {
private:
    std::vector<std::unique_ptr<KinematicNode>> m_nodes;

public:
    KinematicNode* placeholder() {
        auto node = new KinematicNode();
        m_nodes.emplace_back(node);
        return node;
    }

    KinematicNode* add(std::function<KinematicMatrixDual(const fwd::Float& t)> transform) {
        auto node = new KinematicNode(std::move(transform));
        m_nodes.emplace_back(node);
        return node;
    }
};

class Surface {
    private:
    std::string m_mesh_file;
    KinematicNode* m_kinematic_node;
    bool m_lifting;

    public:
    Surface(const std::string& mesh_file, KinematicNode* kinematic_node, bool lifting=true) : m_mesh_file(mesh_file), m_kinematic_node(kinematic_node), m_lifting(lifting) {}
    const std::string& mesh_file() const { return m_mesh_file; }
    bool is_lifting() const { return m_lifting; }
    KinematicNode* kinematics() { return m_kinematic_node; }
};

class Assembly {
    private:
    std::vector<Surface> m_surfaces;
    KinematicNode* m_kinematic_node;

    public:
    Assembly(KinematicNode* kinematic_node) : m_kinematic_node(kinematic_node) {}
    
    void add(const Surface& surface) { 
        m_surfaces.push_back(surface);
    }

    std::vector<Surface>& surfaces() { return m_surfaces; }
    Surface& surface(i64 i) { return m_surfaces.at(i); }
    KinematicNode* kinematics() { return m_kinematic_node; }
};

int main(int /*unused*/, char** /*unused*/) {
    const float u_inf = 1.0f;
    const float amplitude = 0.1f;
    const float omega = 0.5f;

    KinematicsTree kinematics;
    auto body_init = kinematics.add([=](const fwd::Float& t) { return translation_matrix<fwd::Float>({0.0f, 10.0f, 0.0f}); }); // initial position
    auto freestream = kinematics.add([=](const fwd::Float& t) { return translation_matrix<fwd::Float>({-u_inf*t, 0.0f, 0.0f}); })->after(body_init); // freestream
    // auto heave = kinematics.add([=](const fwd::Float& t) { return translation_matrix<fwd::Float>({0.0f, 0.0f, amplitude * fwd::sin(omega * t)}); })->after(freestream); // heave
    auto pitch = kinematics.add([=](const fwd::Float& t) { return rotation_matrix<fwd::Float>({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f * PI_f * fwd::sin(omega * t)); })->after(freestream); // pitch
    
    Assembly assembly(freestream);
    assembly.add(Surface("../../../../mesh/infinite_rectangular_2x2.x", pitch));
    
    auto& wing = assembly.surface(0);

    const f32 t = 1.0f;
    auto vel = wing.kinematics()->linear_velocity(t, {1.0f, 0.0f, 0.0f}); // point velocity
    auto ang_vel = wing.kinematics()->angular_velocity(t);
    std::printf("t: %f, vel: %f %f %f\n", t, vel.x, vel.y, vel.z);
    std::printf("t: %f, ang vel: %f %f %f\n", t, ang_vel.x, ang_vel.y, ang_vel.z);
    
    // CHECK(APPROX(vel.x, -u_inf));
    // CHECK(APPROX(vel.y, 0.0f));
    // CHECK(APPROX(vel.z, amplitude * omega * std::cos(omega * t)));

    CHECK(APPROX(ang_vel.x, 0.0f));
    CHECK(APPROX(ang_vel.y, 0.5f * PI_f * omega * std::cos(omega * t)));
    CHECK(APPROX(ang_vel.z, 0.0f));
    return 0;
}
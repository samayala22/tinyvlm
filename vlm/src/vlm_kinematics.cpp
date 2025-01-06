// vlm_kinematics.cpp
#include "vlm_kinematics.hpp"

namespace vlm {

inline linalg::float4x4 dual_to_float(const KinematicMatrixDual& m) {
    return {
        {m.x.x.val(), m.x.y.val(), m.x.z.val(), m.x.w.val()},
        {m.y.x.val(), m.y.y.val(), m.y.z.val(), m.y.w.val()},
        {m.z.x.val(), m.z.y.val(), m.z.z.val(), m.z.w.val()},
        {m.w.x.val(), m.w.y.val(), m.w.z.val(), m.w.w.val()}
    };
}

KinematicNode::KinematicNode(std::function<KinematicMatrixDual(const fwd::Float& t)> transform)
    : m_transform(std::move(transform)) {}

KinematicNode* KinematicNode::after(KinematicNode* parent) {
    m_parent = parent;
    return this;
}

KinematicMatrixDual KinematicNode::transform_dual(f32 t) const {
    fwd::Float t_dual{t, 1.f};
    auto result = m_transform(t_dual);
    if (m_parent) {
        return linalg::mul(m_parent->transform_dual(t), result);
    }
    return result;
}

KinematicMatrix KinematicNode::transform(f32 t) const {
    return dual_to_float(transform_dual(t));
}

// Note: these should probably be free functions to encourage manual caching
linalg::float3 KinematicNode::linear_velocity(const KinematicMatrixDual& transform_dual, const linalg::float3 vertex) const {
    linalg::vec<fwd::Float, 4> new_pt = linalg::mul(transform_dual, {vertex.x, vertex.y, vertex.z, 1.0f});
    return {new_pt.x.grad(), new_pt.y.grad(), new_pt.z.grad()};
}

linalg::float3 KinematicNode::linear_velocity(f32 t, const linalg::float3 vertex) const {
    return linear_velocity(transform_dual(t), vertex);
}

linalg::float3 KinematicNode::angular_velocity(f32 t) const {
    // Step 1: Get the dual transform
    KinematicMatrixDual dual_transform = transform_dual(t);

    // Step 2: Extract R and R_dot from the dual transform
    // R(t) is the value part, R_dot(t) is the gradient part
    linalg::mat<float, 3, 3> R = {
        {dual_transform.x.x.val(), dual_transform.x.y.val(), dual_transform.x.z.val()},
        {dual_transform.y.x.val(), dual_transform.y.y.val(), dual_transform.y.z.val()},
        {dual_transform.z.x.val(), dual_transform.z.y.val(), dual_transform.z.z.val()}
    };

    linalg::mat<float, 3, 3> R_dot = {
        {dual_transform.x.x.grad(), dual_transform.x.y.grad(), dual_transform.x.z.grad()},
        {dual_transform.y.x.grad(), dual_transform.y.y.grad(), dual_transform.y.z.grad()},
        {dual_transform.z.x.grad(), dual_transform.z.y.grad(), dual_transform.z.z.grad()}
    };

    // Step 3: Compute Omega = R_dot * R^T
    linalg::mat<float, 3, 3> Omega = linalg::mul(R_dot, linalg::transpose(R));

    // Step 4: Extract angular velocity from Omega
    // Omega is skew-symmetric: Omega = [ [0, -wz, wy],
    //                                  [wz, 0, -wx],
    //                                  [-wy, wx, 0] ]
    return {Omega.y.z, Omega.z.x, Omega.x.y};
}

KinematicNode* KinematicsTree::placeholder() {
    auto node = new KinematicNode();
    m_nodes.emplace_back(node);
    return node;
}

KinematicNode* KinematicsTree::add(std::function<KinematicMatrixDual(const fwd::Float& t)> transform) {
    auto node = new KinematicNode(std::move(transform));
    m_nodes.emplace_back(node);
    return node;
}

} // namespace vlm
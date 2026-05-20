#pragma once

#include "vlm_types.hpp"
#include "linalg.h" // Adjust the include path as needed
#include "tinyad.hpp"

#include <functional>
#include <vector>
#include <memory>

namespace vlm {

// Type aliases
template<typename T> using KinematicMatrixDual = linalg::mat<fwd::Number<T>, 4, 4>; /**< Dual kinematic matrix using dual numbers */
template<typename T> using KinematicMatrix = linalg::mat<T, 4, 4>;         /**< Kinematic matrix using floats */

/**
 * @brief Creates a translation matrix for a given translation vector.
 * 
 * @tparam T The data type of the translation vector components.
 * @param translation The translation vector.
 * @return A 4x4 translation matrix.
 */
template <class T>
linalg::mat<T, 4, 4> translation_matrix(const linalg::vec<T, 3>& translation);

/**
 * @brief Creates a skew-symmetric matrix from a given vector.
 * 
 * @tparam T The data type of the vector components.
 * @param a The input vector.
 * @return A 3x3 skew-symmetric matrix.
 */
template <class T>
linalg::mat<T, 3, 3> skew_matrix(const linalg::vec<T, 3>& a);

/**
 * @brief Creates a rotation matrix using Rodrigues' rotation formula.
 * 
 * @tparam T The data type of the matrix components.
 * @param point A point undergoing rotation (used for translation part).
 * @param axis The axis of rotation.
 * @param angle The rotation angle in radians.
 * @return A 4x4 rotation matrix incorporating both rotation and translation.
 */
template <class T>
linalg::mat<T, 4, 4> rotation_matrix(const linalg::vec<T, 3>& point,
                                    const linalg::vec<T, 3>& axis, T angle);

/**
 * @class KinematicNode
 * @brief Represents a node in the kinematics tree, encapsulating a transformation.
 */
template<typename T> // floating point type
class KinematicNode {
private:
    std::function<KinematicMatrixDual<T>(const fwd::Number<T>& t)> m_transform; // transformation function based on dual number time
    KinematicNode<T>* m_parent = nullptr; // pointer to the parent node in the kinematics tree
public:
    /**
     * @brief Constructs a KinematicNode with an optional transformation function.
     * 
     * @param transform A function that takes a dual number time and returns a dual kinematic matrix. Defaults to the identity matrix.
     */
    KinematicNode<T>(std::function<KinematicMatrixDual<T>(const fwd::Number<T>& t)> transform =
                  [](const fwd::Number<T>&) { return linalg::identity; }) : m_transform(std::move(transform)) {};

    /**
     * @brief Sets the parent node for this kinematic node.
     * 
     * @param parent Pointer to the parent KinematicNode.
     * @return A pointer to this KinematicNode for chaining.
     */
    KinematicNode<T>* after(KinematicNode<T>* parent) {
        m_parent = parent;
        return this;
    }

    /**
     * @brief Computes the dual transform matrix at a given time.
     * 
     * This method recursively combines the transform of parent nodes if any.
     * 
     * @param t The time at which to compute the transform.
     * @return The dual transform matrix.
     */
    KinematicMatrixDual<T> transform_dual(T t) const;

    /**
     * @brief Computes the T transform matrix at a given time.
     * @param t The time at which to compute the transform.
     * @return The T transform matrix.
     */
    KinematicMatrix<T> transform(T t) const;

    /**
     * @brief Computes the linear velocity of a vertex at a given time.
     * 
     * @param t The time at which to compute the velocity.
     * @param vertex The vertex position.
     * @return The linear velocity as a 3D T vector.
     */
    linalg::vec<T,3> linear_velocity(T t, const linalg::vec<T,3> vertex) const;
    linalg::vec<T,3> linear_velocity(const KinematicMatrixDual<T>& transform_dual, const linalg::vec<T,3> vertex) const;

    /**
     * @brief Computes the angular velocity of the at a given time.
     * @param t The time at which to compute the angular velocity.
     * @return The angular velocity as a 3D T vector.
     */
    linalg::vec<T,3> angular_velocity(T t) const;
};

/**
 * @class KinematicsTree
 * @brief Manages a collection of KinematicNode objects forming a kinematics tree.
 */
template<typename T> // floating point type
class KinematicsTree {
private:
    std::vector<std::unique_ptr<KinematicNode<T>>> m_nodes; /**< Container for KinematicNode objects */

public:
    /**
     * @brief Creates a placeholder node with the default identity transformation.
     * 
     * @return Pointer to the newly created KinematicNode.
     */
    KinematicNode<T>* placeholder();

    /**
     * @brief Adds a new KinematicNode with a specified transformation function to the tree.
     * 
     * @param transform A function that takes a dual number time and returns a dual kinematic matrix.
     * @return Pointer to the newly added KinematicNode.
     */
    KinematicNode<T>* add(std::function<KinematicMatrixDual<T>(const fwd::Number<T>& t)> transform);
};

// Template function implementations

/**
 * @brief Creates a translation matrix for a given translation vector.
 * 
 * @tparam T The data type of the translation vector components.
 * @param translation The translation vector.
 * @return A 4x4 translation matrix.
 */
template <class T>
linalg::mat<T, 4, 4> translation_matrix(const linalg::vec<T, 3>& translation) {
    return {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {translation, 1}
    };
}

/**
 * @brief Creates a skew-symmetric matrix from a given vector.
 * 
 * @tparam T The data type of the vector components.
 * @param a The input vector.
 * @return A 3x3 skew-symmetric matrix.
 */
template <class T>
linalg::mat<T, 3, 3> skew_matrix(const linalg::vec<T, 3>& a) {
    return {
        {0,     a.z, -a.y},
        {-a.z,  0,     a.x},
        {a.y, -a.x,    0}
    };
}

/**
 * @brief Creates a rotation matrix using Rodrigues' rotation formula.
 * 
 * @tparam T The data type of the matrix components.
 * @param point A point undergoing rotation (used for translation part).
 * @param axis The axis of rotation.
 * @param angle The rotation angle in radians.
 * @return A 4x4 rotation matrix incorporating both rotation and translation.
 */
template <class T>
linalg::mat<T, 4, 4> rotation_matrix(const linalg::vec<T, 3>& point,
                                    const linalg::vec<T, 3>& axis, T angle) {
    using std::sin;
    using std::cos;
    const linalg::mat<T, 3, 3> skew_mat = skew_matrix<T>(axis);
    const linalg::mat<T, 3, 3> identity = linalg::identity;

    const linalg::mat<T, 3, 3> rodrigues = identity + sin(angle) * skew_mat +
        ((T)1.f - cos(angle)) * linalg::mul(skew_mat, skew_mat);
    const linalg::vec<T, 3> trans_part = linalg::mul((identity - rodrigues), point);

    return {
        {rodrigues.x.x, rodrigues.x.y, rodrigues.x.z, 0},
        {rodrigues.y.x, rodrigues.y.y, rodrigues.y.z, 0},
        {rodrigues.z.x, rodrigues.z.y, rodrigues.z.z, 0},
        {trans_part.x, trans_part.y, trans_part.z, 1}
    };
}

template<typename T>
inline linalg::mat<T, 4, 4> dual_to_float(const KinematicMatrixDual<T>& m) {
    return {
        {m.x.x.val(), m.x.y.val(), m.x.z.val(), m.x.w.val()},
        {m.y.x.val(), m.y.y.val(), m.y.z.val(), m.y.w.val()},
        {m.z.x.val(), m.z.y.val(), m.z.z.val(), m.z.w.val()},
        {m.w.x.val(), m.w.y.val(), m.w.z.val(), m.w.w.val()}
    };
}

template<typename T>
KinematicMatrixDual<T> KinematicNode<T>::transform_dual(T t) const {
    fwd::Number<T> t_dual{t, (T)1.f};
    auto result = m_transform(t_dual);
    if (m_parent) {
        return linalg::mul(m_parent->transform_dual(t), result);
    }
    return result;
}

template<typename T>
KinematicMatrix<T> KinematicNode<T>::transform(T t) const {
    return dual_to_float(transform_dual(t));
}

// Note: these should probably be free functions to encourage manual caching
template<typename T>
linalg::vec<T,3> KinematicNode<T>::linear_velocity(const KinematicMatrixDual<T>& transform_dual, const linalg::vec<T,3> vertex) const {
    linalg::vec<fwd::Number<T>, 4> new_pt = linalg::mul(transform_dual, {vertex.x, vertex.y, vertex.z, 1.0f});
    return {new_pt.x.grad(), new_pt.y.grad(), new_pt.z.grad()};
}

template<typename T>
linalg::vec<T,3> KinematicNode<T>::linear_velocity(T t, const linalg::vec<T,3> vertex) const {
    return linear_velocity(transform_dual(t), vertex);
}

template<typename T>
linalg::vec<T,3> KinematicNode<T>::angular_velocity(T t) const {
    // Step 1: Get the dual transform
    KinematicMatrixDual<T> dual_transform = transform_dual(t);

    // Step 2: Extract R and R_dot from the dual transform
    // R(t) is the value part, R_dot(t) is the gradient part
    linalg::mat<T, 3, 3> R = {
        {dual_transform.x.x.val(), dual_transform.x.y.val(), dual_transform.x.z.val()},
        {dual_transform.y.x.val(), dual_transform.y.y.val(), dual_transform.y.z.val()},
        {dual_transform.z.x.val(), dual_transform.z.y.val(), dual_transform.z.z.val()}
    };

    linalg::mat<T, 3, 3> R_dot = {
        {dual_transform.x.x.grad(), dual_transform.x.y.grad(), dual_transform.x.z.grad()},
        {dual_transform.y.x.grad(), dual_transform.y.y.grad(), dual_transform.y.z.grad()},
        {dual_transform.z.x.grad(), dual_transform.z.y.grad(), dual_transform.z.z.grad()}
    };

    // Step 3: Compute Omega = R_dot * R^T
    linalg::mat<T, 3, 3> Omega = linalg::mul(R_dot, linalg::transpose(R));

    // Step 4: Extract angular velocity from Omega
    // Omega is skew-symmetric: Omega = [ [0, -wz, wy],
    //                                  [wz, 0, -wx],
    //                                  [-wy, wx, 0] ]
    return {Omega.y.z, Omega.z.x, Omega.x.y};
}

template<typename T>
KinematicNode<T>* KinematicsTree<T>::placeholder() {
    auto node = new KinematicNode<T>();
    m_nodes.emplace_back(node);
    return node;
}

template<typename T>
KinematicNode<T>* KinematicsTree<T>::add(std::function<KinematicMatrixDual<T>(const fwd::Number<T>& t)> transform) {
    auto node = new KinematicNode<T>(std::move(transform));
    m_nodes.emplace_back(node);
    return node;
}

} // namespace vlm
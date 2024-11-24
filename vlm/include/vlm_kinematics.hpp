#pragma once

#include "vlm_types.hpp"
#include "linalg.h" // Adjust the include path as needed
#include "tinyad.hpp"

#include <functional>
#include <vector>
#include <memory>

namespace vlm {

// Type aliases
using KinematicMatrixDual = linalg::mat<fwd::Float, 4, 4>; /**< Dual kinematic matrix using dual numbers */
using KinematicMatrix = linalg::mat<float, 4, 4>;         /**< Kinematic matrix using floats */

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
class KinematicNode {
private:
    std::function<KinematicMatrixDual(const fwd::Float& t)> m_transform; // transformation function based on dual number time
    KinematicNode* m_parent = nullptr; // pointer to the parent node in the kinematics tree
public:
    /**
     * @brief Constructs a KinematicNode with an optional transformation function.
     * 
     * @param transform A function that takes a dual number time and returns a dual kinematic matrix. Defaults to the identity matrix.
     */
    KinematicNode(std::function<KinematicMatrixDual(const fwd::Float& t)> transform =
                  [](const fwd::Float&) { return linalg::identity; });

    /**
     * @brief Sets the parent node for this kinematic node.
     * 
     * @param parent Pointer to the parent KinematicNode.
     * @return A pointer to this KinematicNode for chaining.
     */
    KinematicNode* after(KinematicNode* parent);

    /**
     * @brief Computes the dual transform matrix at a given time.
     * 
     * This method recursively combines the transform of parent nodes if any.
     * 
     * @param t The time at which to compute the transform.
     * @return The dual transform matrix.
     */
    KinematicMatrixDual transform_dual(float t) const;

    /**
     * @brief Computes the float transform matrix at a given time.
     * @param t The time at which to compute the transform.
     * @return The float transform matrix.
     */
    KinematicMatrix transform(float t) const;

    /**
     * @brief Computes the linear velocity of a vertex at a given time.
     * 
     * @param t The time at which to compute the velocity.
     * @param vertex The vertex position.
     * @return The linear velocity as a 3D float vector.
     */
    linalg::float3 linear_velocity(float t, const linalg::float3 vertex) const;
    linalg::float3 linear_velocity(const KinematicMatrixDual& transform_dual, const linalg::float3 vertex) const;

    /**
     * @brief Computes the angular velocity of the at a given time.
     * @param t The time at which to compute the angular velocity.
     * @return The angular velocity as a 3D float vector.
     */
    linalg::float3 angular_velocity(float t) const;
};

/**
 * @class KinematicsTree
 * @brief Manages a collection of KinematicNode objects forming a kinematics tree.
 */
class KinematicsTree {
private:
    std::vector<std::unique_ptr<KinematicNode>> m_nodes; /**< Container for KinematicNode objects */

public:
    /**
     * @brief Creates a placeholder node with the default identity transformation.
     * 
     * @return Pointer to the newly created KinematicNode.
     */
    KinematicNode* placeholder();

    /**
     * @brief Adds a new KinematicNode with a specified transformation function to the tree.
     * 
     * @param transform A function that takes a dual number time and returns a dual kinematic matrix.
     * @return Pointer to the newly added KinematicNode.
     */
    KinematicNode* add(std::function<KinematicMatrixDual(const fwd::Float& t)> transform);
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
        (1.f - cos(angle)) * linalg::mul(skew_mat, skew_mat);
    const linalg::vec<T, 3> trans_part = linalg::mul((identity - rodrigues), point);

    return {
        {rodrigues.x.x, rodrigues.x.y, rodrigues.x.z, 0},
        {rodrigues.y.x, rodrigues.y.y, rodrigues.y.z, 0},
        {rodrigues.z.x, rodrigues.z.y, rodrigues.z.z, 0},
        {trans_part.x, trans_part.y, trans_part.z, 1}
    };
}

} // namespace vlm
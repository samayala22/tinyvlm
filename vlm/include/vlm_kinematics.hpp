#include "vlm_types.hpp"
#include "tinyad.hpp"
#include "linalg.h"

#include <functional>

namespace vlm {

using tmatrix = linalg::mat<fwd::Float,4,4>;

inline linalg::float4x4 dual_to_float(const tmatrix& m) {
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

    Kinematics& add(std::function<tmatrix(const fwd::Float& t)> joint) {
        m_joints.push_back(std::move(joint));
        return *this;
    }

    tmatrix displacement(float t, i64 n) const {
        fwd::Float t_dual{t, 1.f};
        tmatrix result = linalg::identity;
        for (i64 i = 0; i < n; i++) {
            result = linalg::mul(result, m_joints[i](t_dual));
        }
        return result;
    }
    tmatrix displacement(float t) const {return displacement(t, m_joints.size());}

    linalg::float3 velocity(const tmatrix& transform, const linalg::vec<fwd::Float,4> vertex) const {
        linalg::vec<fwd::Float,4> new_pt = linalg::mul(transform, vertex);
        return {new_pt.x.grad(), new_pt.y.grad(), new_pt.z.grad()};
    }

    std::vector<std::function<tmatrix(const fwd::Float& t)>>& joints() { return m_joints; }

    private:
    std::vector<std::function<tmatrix(const fwd::Float& t)>> m_joints;
    linalg::float4x4 m_transform;
};

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

}
#include "vlm_backend_avx2.hpp"

#include "simpletimer.hpp"
#include "vlm_types.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <immintrin.h>

#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

// #include "mkl.h"

#include <Eigen/Dense>

using namespace vlm;

struct BackendAVX2::linear_solver_t {
    Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXf>> lu;
    linear_solver_t(Eigen::Map<Eigen::MatrixXf>& A) : lu(A) {};
    ~linear_solver_t() = default;
};

BackendAVX2::BackendAVX2(Mesh& mesh, Data& data) : Backend(mesh, data) {
    //tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
    lhs.resize((u64)mesh.nb_panels_wing() * (u64)mesh.nb_panels_wing());
    rhs.resize(mesh.nb_panels_wing());
}

void BackendAVX2::reset() {
    std::fill(data.gamma.begin(), data.gamma.end(), 0.0f);
    std::fill(lhs.begin(), lhs.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);
}

void BackendAVX2::compute_delta_gamma() {
    // Copy the values for the leading edge panels
    for (u32 j = 0; j < mesh.ns; j++) {
        data.delta_gamma[j] = data.gamma[j];
    }
    // note: this is efficient as the memory is contiguous
    for (u32 i = 1; i < mesh.nc; i++) {
        for (u32 j = 0; j < mesh.ns; j++) {
            data.delta_gamma[i*mesh.ns + j] = data.gamma[i*mesh.ns + j] - data.gamma[(i-1)*mesh.ns + j];
        }
    }
}

inline void micro_kernel_influence_scalar(f32& vx, f32& vy, f32& vz, f32& x, f32& y, f32& z, f32& x1, f32& y1, f32& z1, f32& x2, f32& y2, f32& z2) {
    static const f32 rcut = 1.0e-12f;
    vx = 0.0f;
    vy = 0.0f;
    vz = 0.0f;

    // Katz Plotkin, Low speed Aero | Eq 10.115

    const f32 r1r2x =   (y-y1)*(z-z2) - (z-z1)*(y-y2);
    const f32 r1r2y = -((x-x1)*(z-z2) - (z-z1)*(x-x2));
    const f32 r1r2z =   (x-x1)*(y-y2) - (y-y1)*(x-x2);

    const f32 r1 = std::sqrt(pow<2>(x-x1)+pow<2>(y-y1)+pow<2>(z-z1));
    const f32 r2 = std::sqrt(pow<2>(x-x2)+pow<2>(y-y2)+pow<2>(z-z2));
    const f32 square = pow<2>(r1r2x) + pow<2>(r1r2y) + pow<2>(r1r2z);
    if ((r1<rcut) || (r2<rcut) || (square<rcut)) return;

    const f32 r0r1 = (x2-x1)*(x-x1)+(y2-y1)*(y-y1)+(z2-z1)*(z-z1);
    const f32 r0r2 = (x2-x1)*(x-x2)+(y2-y1)*(y-y2)+(z2-z1)*(z-z2);
    const f32 coeff = 1.0f/(4.0f*PI_f*square) * (r0r1/r1 - r0r2/r2);

    vx = coeff * r1r2x;
    vy = coeff * r1r2y;
    vz = coeff * r1r2z;
}

inline void kernel_influence_scalar(f32& inf_x, f32& inf_y, f32& inf_z, f32 x, f32 y, f32 z, f32 x1, f32 y1, f32 z1, f32 x2, f32 y2, f32 z2) {
    f32 vx, vy, vz;
    micro_kernel_influence_scalar(vx, vy, vz, x, y, z, x1, y1, z1, x2, y2, z2);
    inf_x += vx;
    inf_y += vy;
    inf_z += vz;
    y = -y; // wing symmetry
    micro_kernel_influence_scalar(vx, vy, vz, x, y, z, x1, y1, z1, x2, y2, z2);
    inf_x += vx;
    inf_y -= vy;
    inf_z += vz;
}

template<bool Overwrite>
inline void macro_kernel_remainder_scalar(Mesh& m, std::vector<f32>& lhs, u32 ia, u32 lidx) {
    // quick return 
    const u32 remainder =  m.nb_panels_wing() % 8;
    if (remainder == 0) return;

    const u32 v0 = lidx + lidx / m.ns;
    const u32 v1 = v0 + 1;
    const u32 v3 = v0 + m.ns+1;
    const u32 v2 = v3 + 1;
    
    f32 v0x = m.v.x[v0];
    f32 v0y = m.v.y[v0];
    f32 v0z = m.v.z[v0];
    f32 v1x = m.v.x[v1];
    f32 v1y = m.v.y[v1];
    f32 v1z = m.v.z[v1];
    f32 v2x = m.v.x[v2];
    f32 v2y = m.v.y[v2];
    f32 v2z = m.v.z[v2];
    f32 v3x = m.v.x[v3];
    f32 v3y = m.v.y[v3];
    f32 v3z = m.v.z[v3];

    for (u32 ia2 = m.nb_panels_wing() - remainder; ia2 < m.nb_panels_wing(); ia2++) {
        const f32 colloc_x = m.colloc.x[ia2];
        const f32 colloc_y = m.colloc.y[ia2];
        const f32 colloc_z = m.colloc.z[ia2];

        // 3 regs to store induced velocity 
        f32 inf_x = 0.0f;
        f32 inf_y = 0.0f;
        f32 inf_z = 0.0f;
        kernel_influence_scalar(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v0x, v0y, v0z, v1x, v1y, v1z);
        kernel_influence_scalar(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v1x, v1y, v1z, v2x, v2y, v2z);
        kernel_influence_scalar(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v2x, v2y, v2z, v3x, v3y, v3z);
        kernel_influence_scalar(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v3x, v3y, v3z, v0x, v0y, v0z);
        f32 nx = m.normal.x[ia2];
        f32 ny = m.normal.y[ia2];
        f32 nz = m.normal.z[ia2];
        f32 ring_inf = inf_x * nx + inf_y * ny + inf_z * nz;
        // store in col major order
        if (Overwrite) {
            lhs[ia * m.nb_panels_wing() + ia2] = ring_inf;
        } else {
            lhs[ia * m.nb_panels_wing() + ia2] += ring_inf;
        }
    }
}

inline void micro_kernel_influence_avx2(__m256& vx, __m256& vy, __m256& vz, __m256& x, __m256& y, __m256& z, __m256& x1, __m256& y1, __m256& z1, __m256& x2, __m256& y2, __m256& z2, f32 sigma_p4) {
    static const __m256 threshold = _mm256_set1_ps(1.0e-10f);
    static const __m256 four_pi = _mm256_set1_ps(4.0f * PI_f);
    static const __m256 zero = _mm256_set1_ps(0.0f);

    vx = zero;
    vy = zero;
    vz = zero;
    // define vectors
    __m256 r1x = _mm256_sub_ps(x, x1);
    __m256 r1y = _mm256_sub_ps(y, y1);
    __m256 r1z = _mm256_sub_ps(z, z1);
    __m256 r2x = _mm256_sub_ps(x, x2);
    __m256 r2y = _mm256_sub_ps(y, y2);
    __m256 r2z = _mm256_sub_ps(z, z2);

    // crossproduct
    // (v0y*v1z - v0z*v1y);
    // (v0z*v1x - v0x*v1z);
    // (v0x*v1y - v0y*v1x);
    __m256 r1_x_r2_x = _mm256_fmsub_ps(r1y, r2z, _mm256_mul_ps(r1z, r2y));
    __m256 r1_x_r2_y = _mm256_fmsub_ps(r1z, r2x, _mm256_mul_ps(r1x, r2z));
    __m256 r1_x_r2_z = _mm256_fmsub_ps(r1x, r2y, _mm256_mul_ps(r1y, r2x));

    // magnitude & mag squared of crossproduct
    __m256 r1_x_r2_mag_p2 = _mm256_fmadd_ps(r1_x_r2_x, r1_x_r2_x, _mm256_fmadd_ps(r1_x_r2_y, r1_x_r2_y, _mm256_mul_ps(r1_x_r2_z, r1_x_r2_z)));
    __m256 r1_mag = _mm256_sqrt_ps(_mm256_fmadd_ps(r1x, r1x, _mm256_fmadd_ps(r1y, r1y, _mm256_mul_ps(r1z, r1z))));
    __m256 r2_mag = _mm256_sqrt_ps(_mm256_fmadd_ps(r2x, r2x, _mm256_fmadd_ps(r2y, r2y, _mm256_mul_ps(r2z, r2z))));

    // vector from point 1 to point 2 of segment
    __m256 r0x = _mm256_sub_ps(x2, x1);
    __m256 r0y = _mm256_sub_ps(y2, y1);
    __m256 r0z = _mm256_sub_ps(z2, z1);

    // dot product r0.r1 and r0.r2
    __m256 r0_dot_r1 = _mm256_fmadd_ps(r0x, r1x, _mm256_fmadd_ps(r0y, r1y, _mm256_mul_ps(r0z, r1z)));
    __m256 r0_dot_r2 = _mm256_fmadd_ps(r0x, r2x, _mm256_fmadd_ps(r0y, r2y, _mm256_mul_ps(r0z, r2z)));

    __m256 numerator = _mm256_fmsub_ps(r0_dot_r1, r2_mag, _mm256_mul_ps(r0_dot_r2, r1_mag));

    __m256 four_pi_r1_mag_r2_mag = _mm256_mul_ps(four_pi, _mm256_mul_ps(r1_mag, r2_mag)); // 4*pi*|r1|*|r2|
    __m256 denominator;

    if (sigma_p4 == 0.0f) {
        // Singular Bio-Savart
        denominator = _mm256_mul_ps(four_pi_r1_mag_r2_mag, r1_x_r2_mag_p2);
    } else {
        // Vatistas smoothing kernel (n=2) (https://doi.org/10.3390/fluids7020081)
        __m256 r1_x_r2_mag_p4 = _mm256_mul_ps(r1_x_r2_mag_p2, r1_x_r2_mag_p2); // ^2n
        __m256 r0_mag_p2 = _mm256_fmadd_ps(r0x, r0x, _mm256_fmadd_ps(r0y, r0y, _mm256_mul_ps(r0z, r0z)));
        __m256 r0_mag_p4 = _mm256_mul_ps(r0_mag_p2, r0_mag_p2); // ^2n
        denominator = _mm256_mul_ps(four_pi_r1_mag_r2_mag, _mm256_sqrt_ps(_mm256_fmadd_ps(_mm256_set1_ps(sigma_p4), r0_mag_p4, r1_x_r2_mag_p4)));
    }
    
    __m256 coeff = _mm256_div_ps(numerator, denominator);

    // add the influence and blend with mask
    // the masks should be done independently for optimal ILP but if compiler smart he can do it
    __m256 mask = _mm256_cmp_ps(r1_mag, threshold, _CMP_LT_OS);
    mask = _mm256_or_ps(mask, _mm256_cmp_ps(r2_mag, threshold, _CMP_LT_OS));
    mask = _mm256_or_ps(mask, _mm256_cmp_ps(r1_x_r2_mag_p2, threshold, _CMP_LT_OS));

    vx = _mm256_blendv_ps(_mm256_mul_ps(r1_x_r2_x, coeff), zero, mask);
    vy = _mm256_blendv_ps(_mm256_mul_ps(r1_x_r2_y, coeff), zero, mask);
    vz = _mm256_blendv_ps(_mm256_mul_ps(r1_x_r2_z, coeff), zero, mask);
}

inline void kernel_influence_avx2(__m256& inf_x, __m256& inf_y, __m256& inf_z, __m256 x, __m256 y, __m256 z, __m256 x1, __m256 y1, __m256 z1, __m256 x2, __m256 y2, __m256 z2, f32 sigma_p4) {
    __m256 vx, vy, vz;
    micro_kernel_influence_avx2(vx, vy, vz, x, y, z, x1, y1, z1, x2, y2, z2, sigma_p4);
    inf_x = _mm256_add_ps(inf_x, vx);
    inf_y = _mm256_add_ps(inf_y, vy);
    inf_z = _mm256_add_ps(inf_z, vz);
    y = _mm256_xor_ps(y, _mm256_set1_ps(-0.0f)); // wing symmetry
    micro_kernel_influence_avx2(vx, vy, vz, x, y, z, x1, y1, z1, x2, y2, z2, sigma_p4);
    inf_x = _mm256_add_ps(inf_x, vx);
    inf_y = _mm256_sub_ps(inf_y, vy); // HERE IS SUB INSTEAD OF ADD !
    inf_z = _mm256_add_ps(inf_z, vz);
}

template<bool Overwrite>
inline void macro_kernel_avx2(Mesh& m, std::vector<f32>& lhs, u32 ia, u32 lidx, f32 sigma_p4) {
    const u32 v0 = lidx + lidx / m.ns;
    const u32 v1 = v0 + 1;
    const u32 v3 = v0 + m.ns+1;
    const u32 v2 = v3 + 1;
    // in reality only load v1 & v2 and reuse data from previous v1 & v2 to be new v0 & v3 respectively
    // 12 regs (6 loads + 6 reuse) -> these will get spilled once we get in the influence function
    __m256 v0x = _mm256_broadcast_ss(&m.v.x[v0]);
    __m256 v0y = _mm256_broadcast_ss(&m.v.y[v0]);
    __m256 v0z = _mm256_broadcast_ss(&m.v.z[v0]);
    __m256 v1x = _mm256_broadcast_ss(&m.v.x[v1]);
    __m256 v1y = _mm256_broadcast_ss(&m.v.y[v1]);
    __m256 v1z = _mm256_broadcast_ss(&m.v.z[v1]);
    __m256 v2x = _mm256_broadcast_ss(&m.v.x[v2]);
    __m256 v2y = _mm256_broadcast_ss(&m.v.y[v2]);
    __m256 v2z = _mm256_broadcast_ss(&m.v.z[v2]);
    __m256 v3x = _mm256_broadcast_ss(&m.v.x[v3]);
    __m256 v3y = _mm256_broadcast_ss(&m.v.y[v3]);
    __m256 v3z = _mm256_broadcast_ss(&m.v.z[v3]);

    for (u32 ia2 = 0; ia2 <= m.nb_panels_wing()-8; ia2+=8) {
        // loads (3 regs)
        __m256 colloc_x = _mm256_loadu_ps(&m.colloc.x[ia2]);
        __m256 colloc_y = _mm256_loadu_ps(&m.colloc.y[ia2]);
        __m256 colloc_z = _mm256_loadu_ps(&m.colloc.z[ia2]);
        // 3 regs to store induced velocity
        __m256 inf_x = _mm256_setzero_ps();
        __m256 inf_y = _mm256_setzero_ps();
        __m256 inf_z = _mm256_setzero_ps();

        kernel_influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v0x, v0y, v0z, v1x, v1y, v1z, sigma_p4);
        kernel_influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v1x, v1y, v1z, v2x, v2y, v2z, sigma_p4);
        kernel_influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v2x, v2y, v2z, v3x, v3y, v3z, sigma_p4);
        kernel_influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v3x, v3y, v3z, v0x, v0y, v0z, sigma_p4);

        // dot product
        __m256 nx = _mm256_loadu_ps(&m.normal.x[ia2]);
        __m256 ny = _mm256_loadu_ps(&m.normal.y[ia2]);
        __m256 nz = _mm256_loadu_ps(&m.normal.z[ia2]);
        __m256 ring_inf = _mm256_fmadd_ps(inf_x, nx, _mm256_fmadd_ps(inf_y, ny, _mm256_mul_ps(inf_z, nz)));

        // store in col major order
        if (Overwrite) {
            _mm256_storeu_ps(&lhs[ia * m.nb_panels_wing() + ia2], ring_inf);
        } else {                
            __m256 lhs_ia = _mm256_loadu_ps(&lhs[ia * m.nb_panels_wing() + ia2]);
            lhs_ia = _mm256_add_ps(lhs_ia, ring_inf);
            _mm256_storeu_ps(&lhs[ia * m.nb_panels_wing() + ia2], lhs_ia);
        }
    }
}

void BackendAVX2::compute_lhs() {
    SimpleTimer timer("LHS");
    Mesh& m = mesh;
    const f32 sigma_p4 = pow<4>(data.sigma_vatistas); // Vatistas coeffcient (^2n with n=2)
    tbb::affinity_partitioner ap;

    const u32 start_wing = 0;
    const u32 end_wing = (m.nc - 1) * m.ns;
    tbb::parallel_for(tbb::blocked_range<u32>(start_wing, end_wing),[&](const tbb::blocked_range<u32> &r) {
    for (u32 i = r.begin(); i < r.end(); i++) {
        macro_kernel_avx2<true>(m, lhs, i, i, sigma_p4);
        macro_kernel_remainder_scalar<true>(m, lhs, i, i);
    }
    }, ap);

    for (u32 i = m.nc - 1; i < m.nc + m.nw; i++) {
        tbb::parallel_for(tbb::blocked_range<u32>(0, m.ns),[&](const tbb::blocked_range<u32> &r) {
        for (u32 j = r.begin(); j < r.end(); j++) {
            const u32 ia = (m.nc - 1) * m.ns + j;
            const u32 lidx = i * m.ns + j;
            macro_kernel_avx2<false>(m, lhs, ia, lidx, sigma_p4);
            macro_kernel_remainder_scalar<false>(m, lhs, i, i);
        }
        }, ap);
    }
}

void BackendAVX2::compute_rhs() {
    SimpleTimer timer("RHS");
    Data& d = data;
    Mesh& m = mesh;
    for (u32 i = 0; i < mesh.nb_panels_wing(); i++) {
        rhs[i] = - (d.u_inf.x() * m.normal.x[i] + d.u_inf.y() * m.normal.y[i] + d.u_inf.z() * m.normal.z[i]);
    }
}

void BackendAVX2::rebuild_rhs(const std::vector<f32>& section_alphas) {

}

// void BackendAVX2::lu_solve() {
//     SimpleTimer timer("Solve");
//     const MKL_INT n = static_cast<MKL_INT>(mesh.nb_panels_wing());

//     std::vector<MKL_INT> ipiv(n);

//     // Use LAPACK_COL_MAJOR since the data is stored in column-major format
//     MKL_INT info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, lhs.data(), n, ipiv.data());
//     if (info != 0) {
//         throw std::runtime_error("Failed to compute lhs matrix");
//     }

//     info = LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, 1, lhs.data(), n, ipiv.data(), rhs.data(), n);
//     if (info != 0) {
//         throw std::runtime_error("Failed to solve linear system");
//     }

//     // Copy the solution from rhs to data.gamma
//     std::copy(rhs.begin(), rhs.end(), data.gamma.begin());
// }

void BackendAVX2::lu_factor() {
    SimpleTimer timer("Factor");
    const u32 n = mesh.nb_panels_wing();
    Eigen::Map<Eigen::MatrixXf> A(lhs.data(), n, n);
    solver = std::make_unique<linear_solver_t>(A);
}

void BackendAVX2::lu_solve() {
    SimpleTimer timer("Solve");
    const u32 n = mesh.nb_panels_wing();
    Eigen::Map<Eigen::VectorXf> x(data.gamma.data(), n);
    Eigen::Map<Eigen::VectorXf> b(rhs.data(), n);
    
    x = solver->lu.solve(b);
}

/// @brief Compute some helping variables for the cm computation
/// @param m Mesh object
/// @param d Data object
/// @param dl Leading edge vector pointing outward from wing root
/// @param dst_to_ref Distance from the center of leading edge to the reference point
/// @param i Panel index
inline void compute_panel_vectors(const Mesh& m, const Data& d, Eigen::Vector3f& dl, Eigen::Vector3f& dst_to_ref, u32 i) {
    Eigen::Vector3f v0 = m.get_v0(i);
    Eigen::Vector3f v1 = m.get_v1(i);
    // Leading edge vector pointing outward from wing root
    dl = v1 - v0;
    // Distance from the center of leading edge to the reference point
    dst_to_ref = d.ref_pt - 0.5f * (v0 + v1);
}

inline Eigen::Vector3f compute_panel_forces(const Mesh& m, const Data& d, const Eigen::Vector3f& dl, u32 i) {
    // force = U_inf x (panel_dl * rho * delta_gamma)
    return d.u_inf.cross(dl) * d.rho * d.delta_gamma[i];
}

f32 BackendAVX2::compute_coefficient_cl(const Mesh& mesh, const Data& data, const u32 j, const u32 n, const f32 area) {
    f32 cl = 0.0f;
    for (u32 u = 0; u < mesh.nc; u++) {
        for (u32 v = j; v < j + n; v++) {
            const u32 li = u * mesh.ns + v; // linear index
            const Eigen::Vector3f v0 = mesh.get_v0(li);
            const Eigen::Vector3f v1 = mesh.get_v1(li);
            // Leading edge vector pointing outward from wing root
            const Eigen::Vector3f dl = v1 - v0;
            // Distance from the center of leading edge to the reference point
            const Eigen::Vector3f force = data.u_inf.cross(dl) * data.rho * data.delta_gamma[li];
            cl += force.dot(data.lift_axis);
        }
    }
    cl /= 0.5f * data.rho * data.u_inf.squaredNorm() * area;

    return cl;
}

Eigen::Vector3f BackendAVX2::compute_coefficient_cm(const Mesh& mesh, const Data& data, const u32 j, const u32 n, const f32 area, const f32 chord) {
    Eigen::Vector3f cm = Eigen::Vector3f::Zero();
    for (u32 u = 0; u < mesh.nc; u++) {
        for (u32 v = j; v < j + n; v++) {
            const u32 li = u * mesh.ns + v; // linear index
            const Eigen::Vector3f v0 = mesh.get_v0(li);
            const Eigen::Vector3f v1 = mesh.get_v1(li);
            // Leading edge vector pointing outward from wing root
            const Eigen::Vector3f dl = v1 - v0;
            // Distance from the center of leading edge to the reference point
            const Eigen::Vector3f dst_to_ref = data.ref_pt - 0.5f * (v0 + v1);
            // Distance from the center of leading edge to the reference point
            const Eigen::Vector3f force = data.u_inf.cross(dl) * data.rho * data.delta_gamma[li];
            cm += force.cross(dst_to_ref);
        }
    }
    cm /= 0.5f * data.rho * data.u_inf.squaredNorm() * area * chord;
    return cm;
}

f32 BackendAVX2::compute_coefficient_cd(const Mesh& mesh, const Data& data, const u32 j, const u32 n, const f32 area) {
    assert(n > 0);
    assert(j > 0 and j+n <= mesh.ns);

    f32 cd = 0.0f;
    // Drag coefficent computed using Trefftz plane
    const u32 begin = j + mesh.nb_panels_wing();
    const u32 end = begin + n;
    // parallel for
    for (u32 ia = begin; ia < end; ia++) {
        const f32 colloc_x = mesh.colloc.x[ia];
        const f32 colloc_y = mesh.colloc.y[ia];
        const f32 colloc_z = mesh.colloc.z[ia];
        Eigen::Vector3f inf = Eigen::Vector3f::Zero();
        for (u32 ia2 = begin; ia2 < end; ia2++) {
            Eigen::Vector3f inf2 = Eigen::Vector3f::Zero();
            Eigen::Vector3f v0 = mesh.get_v0(ia2);
            Eigen::Vector3f v1 = mesh.get_v1(ia2);
            Eigen::Vector3f v2 = mesh.get_v2(ia2);
            Eigen::Vector3f v3 = mesh.get_v3(ia2);
            // Influence from the streamwise vortex lines
            kernel_influence_scalar(inf2.x(), inf2.y(), inf2.z(), colloc_x, colloc_y, colloc_z, v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z());
            kernel_influence_scalar(inf2.x(), inf2.y(), inf2.z(), colloc_x, colloc_y, colloc_z, v3.x(), v3.y(), v3.z(), v0.x(), v0.y(), v0.z());
            f32 gamma_w = data.gamma[(mesh.nc-1)*mesh.ns + ia2 % mesh.ns];
            // This is the induced velocity calculated with the vortex (gamma) calculated earlier (according to kutta condition)
            inf += gamma_w * inf2;
        }
        const Eigen::Vector3f normal{mesh.normal.x[ia], mesh.normal.y[ia], mesh.normal.z[ia]};
        const f32 w_ind = inf.dot(normal);
        const u32 col = ia % mesh.ns;
        Eigen::Vector3f v0 = mesh.get_v0(ia);
        Eigen::Vector3f v1 = mesh.get_v1(ia);
        const f32 dl = (v1 - v0).norm();
        cd -= 0.5f * data.rho * data.gamma[(mesh.nc-1)*mesh.ns + col] * w_ind * dl;
    }
    cd /= 0.5f * data.rho * data.u_inf.squaredNorm() * area;
    return cd;
}

void BackendAVX2::compute_coefficients() {
    Mesh& m = mesh;
    SimpleTimer timer("Compute coefficients");

    for (u32 i = 0; i < mesh.nb_panels_wing(); i++) {
        Eigen::Vector3f dl, dst_to_ref;
        compute_panel_vectors(m, data, dl, dst_to_ref, i);
        Eigen::Vector3f force = compute_panel_forces(m, data, dl, i);
        data.cl += force.dot(data.lift_axis);
        data.cm += force.cross(dst_to_ref);
    }

    // Drag coefficent computed using Trefftz plane
    for (u32 ia = mesh.nb_panels_wing(); ia < mesh.nb_panels_total(); ia++) {
        const f32 colloc_x = mesh.colloc.x[ia];
        const f32 colloc_y = mesh.colloc.y[ia];
        const f32 colloc_z = mesh.colloc.z[ia];
        Eigen::Vector3f inf = Eigen::Vector3f::Zero();
        for (u32 ia2 = mesh.nb_panels_wing(); ia2 < mesh.nb_panels_total(); ia2++) {
            Eigen::Vector3f inf2 = Eigen::Vector3f::Zero();
            Eigen::Vector3f v0 = mesh.get_v0(ia2);
            Eigen::Vector3f v1 = mesh.get_v1(ia2);
            Eigen::Vector3f v2 = mesh.get_v2(ia2);
            Eigen::Vector3f v3 = mesh.get_v3(ia2);
            // Influence from the streamwise vortex lines
            kernel_influence_scalar(inf2.x(), inf2.y(), inf2.z(), colloc_x, colloc_y, colloc_z, v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z());
            kernel_influence_scalar(inf2.x(), inf2.y(), inf2.z(), colloc_x, colloc_y, colloc_z, v3.x(), v3.y(), v3.z(), v0.x(), v0.y(), v0.z());
            f32 gamma_w = data.gamma[(m.nc-1)*m.ns + ia2 % m.ns];
            // This is the induced velocity calculated with the vortex (gamma) calculated earlier (according to kutta condition)
            inf += gamma_w * inf2;
        }
        const Eigen::Vector3f normal{m.normal.x[ia], m.normal.y[ia], m.normal.z[ia]};
        const f32 w_ind = inf.dot(normal);
        const u32 col = ia % m.ns;
        Eigen::Vector3f v0 = mesh.get_v0(ia);
        Eigen::Vector3f v1 = mesh.get_v1(ia);
        const f32 dl = (v1 - v0).norm();
        data.cd -= 0.5f * data.rho * data.gamma[(m.nc-1)*m.ns + col] * w_ind * dl;
    }

    const f32 common = 0.5f * data.rho * data.u_inf.squaredNorm() * data.s_ref;
    data.cl /= common;
    data.cm /= common * data.c_ref;
    data.cd /= common;

    // Cd = Cl^2 / (pi * AR * e) with AR = b^2 / S_ref
    // std::cout << "CD analytic: " << 2.0f * (data.cl * data.cl) / (PI_f * (10.0f*10.0f / data.s_ref)) << std::endl;
}
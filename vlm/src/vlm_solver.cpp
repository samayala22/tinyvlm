#include "vlm_data.hpp"
#include "vlm_config.hpp"
#include "vlm_io.hpp"
#include "vlm_mesh.hpp"

#include "vlm_solver.hpp"
#include "vlm_types.hpp"

#include "simpletimer.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <immintrin.h>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>

using namespace vlm;

Solver::Solver(Mesh& mesh, Data& data, IO& io, Config& config) : mesh(mesh), data(data), io(io), config(config) {
    lhs.resize(mesh.nb_panels_wing() * mesh.nb_panels_wing());
    rhs.resize(mesh.nb_panels_wing());
}

void Solver::reset() {
    std::fill(data.gamma.begin(), data.gamma.end(), 0.0f);
    std::fill(lhs.begin(), lhs.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);
}

void Solver::run(const f32 alpha) {
    SimpleTimer timer("SOLVER RUN");
    reset();
    data.reset();
    data.compute_freestream(alpha);
    mesh.update_wake(data.u_inf);
    compute_lhs();
    compute_rhs();
    solve();
    data.postprocess();
    compute_forces();

    std::cout << "Alpha: " << alpha << " CL: " << data.cl << " CD: " << data.cd << " CM: " << data.cm << std::endl;
}

inline void influence(Vec3& inf, f32 x, f32 y, f32 z, f32 x1, f32 y1, f32 z1, f32 x2, f32 y2, f32 z2) {
    static const f32 rcut = 1.0e-12f;

    {
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

        inf.x += coeff * r1r2x;
        inf.y += coeff * r1r2y;
        inf.z += coeff * r1r2z;
    }

    // wing symmetry
    y = -y;

    {
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

        inf.x += coeff * r1r2x;
        inf.y -= coeff * r1r2y;
        inf.z += coeff * r1r2z;
    }
}


// void Solver::compute_lhs() {
//     // row major order
//     SimpleTimer timer("LHS");
//     Mesh& m = mesh;
//     // influence matrix is row major
//     // loop over rows
//     for (u32 ni = 0; ni < m.nc; ni++) {
//         for (u32 nj = 0; nj < m.ns; nj++) {
//             const u32 i = ni * m.ns + nj;
//             const f32 colloc_x = m.colloc.x[i];
//             const f32 colloc_y = m.colloc.y[i];
//             const f32 colloc_z = m.colloc.z[i];

//             for (u32 i2 = 0; i2 < m.nc; i2++) {
//                 for (u32 j2 = 0; j2 < m.ns; j2++) {
//                     const u32 ia2 = i2 * m.ns + j2;
//                     Vec3 inf;
//                     // Influence from the 4 edges
//                     influence(inf, colloc_x, colloc_y, colloc_z, m.v0.x[ia2], m.v0.y[ia2], m.v0.z[ia2], m.v1.x[ia2], m.v1.y[ia2], m.v1.z[ia2]);
//                     influence(inf, colloc_x, colloc_y, colloc_z, m.v1.x[ia2], m.v1.y[ia2], m.v1.z[ia2], m.v2.x[ia2], m.v2.y[ia2], m.v2.z[ia2]);
//                     influence(inf, colloc_x, colloc_y, colloc_z, m.v2.x[ia2], m.v2.y[ia2], m.v2.z[ia2], m.v3.x[ia2], m.v3.y[ia2], m.v3.z[ia2]);
//                     influence(inf, colloc_x, colloc_y, colloc_z, m.v3.x[ia2], m.v3.y[ia2], m.v3.z[ia2], m.v0.x[ia2], m.v0.y[ia2], m.v0.z[ia2]);

//                     if (i2 == m.nc - 1) {
//                         // TODO: this should be a loop over the wake panels in chordwise direction
//                         const u32 w = m.nc * m.ns + j2;
//                         influence(inf, colloc_x, colloc_y, colloc_z, m.v0.x[w], m.v0.y[w], m.v0.z[w], m.v1.x[w], m.v1.y[w], m.v1.z[w]);
//                         influence(inf, colloc_x, colloc_y, colloc_z, m.v1.x[w], m.v1.y[w], m.v1.z[w], m.v2.x[w], m.v2.y[w], m.v2.z[w]);
//                         influence(inf, colloc_x, colloc_y, colloc_z, m.v2.x[w], m.v2.y[w], m.v2.z[w], m.v3.x[w], m.v3.y[w], m.v3.z[w]);
//                         influence(inf, colloc_x, colloc_y, colloc_z, m.v3.x[w], m.v3.y[w], m.v3.z[w], m.v0.x[w], m.v0.y[w], m.v0.z[w]);
//                     }

//                     lhs[i * m.nb_panels_wing() + ia2] = inf.x * m.normal.x[i] + inf.y * m.normal.y[i] + inf.z * m.normal.z[i];
//                 }
//             }
//         }
//     }

//     // std::ofstream f("lhs.txt");
//     // for (u32 i = 0; i < m.nb_panels_wing(); i++) {
//     //     for (u32 j = 0; j < m.nb_panels_wing(); j++) {
//     //         f << lhs[i * m.nb_panels_wing() + j] << "\n";
//     //     }
//     // }
//     // f.close();
// }

// void Solver::compute_lhs() {
//     SimpleTimer timer("LHS");
//     Mesh& m = mesh;
//     const u32 v_ns = m.ns + 1;

//     for (u32 i = 0; i < m.nc - 1; i++) {
//         for (u32 j = 0; j < m.ns; j++) {
//             const u32 ia = i * m.ns + j;
//             const u32 v0 = i * v_ns + j;
//             const u32 v1 = v0 + 1;
//             const u32 v3 = (i + 1) * v_ns + j;
//             const u32 v2 = v3 + 1;
//             // in reality only load v1 & v2 and reuse data from previous v1 & v2 to be new v0 & v3 respectively
//             // 12 regs (6 loads + 6 reuse) -> these will get spilled once we get in the influence function
//             f32 v0x = m.v.x[v0];
//             f32 v0y = m.v.y[v0];
//             f32 v0z = m.v.z[v0];
//             f32 v1x = m.v.x[v1];
//             f32 v1y = m.v.y[v1];
//             f32 v1z = m.v.z[v1];
//             f32 v2x = m.v.x[v2];
//             f32 v2y = m.v.y[v2];
//             f32 v2z = m.v.z[v2];
//             f32 v3x = m.v.x[v3];
//             f32 v3y = m.v.y[v3];
//             f32 v3z = m.v.z[v3];
//             for (u32 ia2 = 0; ia2 < m.nb_panels_wing(); ia2++) {
//                 if (ia == 0 && ia2 == 8) {
//                     for (u32 k = 0; k < 8; k++) std::cout << lhs[k] << std::endl;
//                 }
//                 // loads (3 regs)
//                 const f32 colloc_x = m.colloc.x[ia2];
//                 const f32 colloc_y = m.colloc.y[ia2];
//                 const f32 colloc_z = m.colloc.z[ia2];

//                 // 3 regs to store induced velocity 
//                 Vec3 inf;
//                 influence(inf, colloc_x, colloc_y, colloc_z, v0x, v0y, v0z, v1x, v1y, v1z);
//                 influence(inf, colloc_x, colloc_y, colloc_z, v1x, v1y, v1z, v2x, v2y, v2z);
//                 influence(inf, colloc_x, colloc_y, colloc_z, v2x, v2y, v2z, v3x, v3y, v3z);
//                 influence(inf, colloc_x, colloc_y, colloc_z, v3x, v3y, v3z, v0x, v0y, v0z);

//                 // storing in col major order
//                 lhs[ia * m.nb_panels_wing() + ia2] = inf.x * m.normal.x[ia2] + inf.y * m.normal.y[ia2] + inf.z * m.normal.z[ia2];
//             }
//         }
//     }

//     for (u32 i = m.nc - 1; i < m.nc + m.nw; i++) {
//         for (u32 j = 0; j < m.ns; j++) {
//             const u32 ia = (m.nc - 1) * m.ns + j;
//             const u32 v0 = i * v_ns + j;
//             const u32 v1 = v0 + 1;
//             const u32 v3 = (i + 1) * v_ns + j;
//             const u32 v2 = v3 + 1;
//             // in reality only load v1 & v2 and reuse data from previous v1 & v2 to be new v0 & v3 respectively
//             // 12 regs (6 loads + 6 reuse) -> these will get spilled once we get in the influence function
//             f32 v0x = m.v.x[v0];
//             f32 v0y = m.v.y[v0];
//             f32 v0z = m.v.z[v0];
//             f32 v1x = m.v.x[v1];
//             f32 v1y = m.v.y[v1];
//             f32 v1z = m.v.z[v1];
//             f32 v2x = m.v.x[v2];
//             f32 v2y = m.v.y[v2];
//             f32 v2z = m.v.z[v2];
//             f32 v3x = m.v.x[v3];
//             f32 v3y = m.v.y[v3];
//             f32 v3z = m.v.z[v3];
//             for (u32 ia2 = 0; ia2 < m.nb_panels_wing(); ia2++) {
//                 // loads (3 regs)
//                 const f32 colloc_x = m.colloc.x[ia2];
//                 const f32 colloc_y = m.colloc.y[ia2];
//                 const f32 colloc_z = m.colloc.z[ia2];

//                 // 3 regs to store induced velocity 
//                 Vec3 inf;
//                 influence(inf, colloc_x, colloc_y, colloc_z, v0x, v0y, v0z, v1x, v1y, v1z);
//                 influence(inf, colloc_x, colloc_y, colloc_z, v1x, v1y, v1z, v2x, v2y, v2z);
//                 influence(inf, colloc_x, colloc_y, colloc_z, v2x, v2y, v2z, v3x, v3y, v3z);
//                 influence(inf, colloc_x, colloc_y, colloc_z, v3x, v3y, v3z, v0x, v0y, v0z);

//                 // storing in col major order
//                 lhs[ia * m.nb_panels_wing() + ia2] += inf.x * m.normal.x[ia2] + inf.y * m.normal.y[ia2] + inf.z * m.normal.z[ia2];
//             }
//         }
//     }
// }

inline void influence_avx2(__m256& inf_x, __m256& inf_y, __m256& inf_z, __m256 x, __m256 y, __m256 z, __m256 x1, __m256 y1, __m256 z1, __m256 x2, __m256 y2, __m256 z2) {
    // r cutoff
    static const __m256 threshold = _mm256_set1_ps(1.0e-12f);
    static const __m256 pi4 = _mm256_set1_ps(4.0f * PI_f);
    static const __m256 zero = _mm256_set1_ps(0.0f);
    {
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
        __m256 r1r2x = _mm256_fmsub_ps(r1y, r2z, _mm256_mul_ps(r1z, r2y));
        __m256 r1r2y = _mm256_fmsub_ps(r1z, r2x, _mm256_mul_ps(r1x, r2z));
        __m256 r1r2z = _mm256_fmsub_ps(r1x, r2y, _mm256_mul_ps(r1y, r2x));
        // asuming that the compiler is smart enough to reuse the previous registers
        __m256 r1r2x_sq = _mm256_mul_ps(r1r2x, r1r2x);
        __m256 r1r2y_sq = _mm256_mul_ps(r1r2y, r1r2y);
        __m256 r1r2z_sq = _mm256_mul_ps(r1r2z, r1r2z);

        // magnitude & mag squared of crossproduct
        __m256 square = _mm256_add_ps(r1r2x_sq, _mm256_add_ps(r1r2y_sq, r1r2z_sq));
        __m256 r1 = _mm256_sqrt_ps(_mm256_fmadd_ps(r1x, r1x, _mm256_fmadd_ps(r1y, r1y, _mm256_mul_ps(r1z, r1z))));
        __m256 r2 = _mm256_sqrt_ps(_mm256_fmadd_ps(r2x, r2x, _mm256_fmadd_ps(r2y, r2y, _mm256_mul_ps(r2z, r2z))));

        // (r1 - r2) x [r1 / |r1| - r2 / |r2|]
        // = (r1 x r2) * (|r2| - |r1|) / (|r1| * |r2|)  (simplification)
        // so lets define the scalar "factor" = (|r2| - |r1|) / (|r1| * |r2|)
        __m256 factor = _mm256_div_ps(_mm256_sub_ps(r2, r1), _mm256_mul_ps(r1, r2));

        // induced velocity (assume unit strength vortex)
        // v = (r1 x r2)^2 * factor / (4 * pi * |(r1 x r2)|^2)
        // v = (r1 x r2)^2 * coeff
        __m256 coeff = _mm256_div_ps(factor, _mm256_mul_ps(pi4, square));

        // add the influence and blend with mask
        // the masks should be done independently for optimal ILP but if compiler smart he can do it
        __m256 mask = _mm256_cmp_ps(r1, threshold, _CMP_LT_OS);
        mask = _mm256_or_ps(mask, _mm256_cmp_ps(r2, threshold, _CMP_LT_OS));
        mask = _mm256_or_ps(mask, _mm256_cmp_ps(square, threshold, _CMP_LT_OS));

        inf_x = _mm256_add_ps(inf_x, _mm256_blendv_ps(_mm256_mul_ps(r1r2x_sq, coeff), zero, mask));
        inf_y = _mm256_add_ps(inf_y, _mm256_blendv_ps(_mm256_mul_ps(r1r2y_sq, coeff), zero, mask));
        inf_z = _mm256_add_ps(inf_z, _mm256_blendv_ps(_mm256_mul_ps(r1r2z_sq, coeff), zero, mask));
    }

    // wing symmetry
    y = _mm256_xor_ps(y, _mm256_set1_ps(-0.0f));

    {
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
        __m256 r1r2x = _mm256_fmsub_ps(r1y, r2z, _mm256_mul_ps(r1z, r2y));
        __m256 r1r2y = _mm256_fmsub_ps(r1z, r2x, _mm256_mul_ps(r1x, r2z));
        __m256 r1r2z = _mm256_fmsub_ps(r1x, r2y, _mm256_mul_ps(r1y, r2x));
        // asuming that the compiler is smart enough to reuse the previous registers
        __m256 r1r2x_sq = _mm256_mul_ps(r1r2x, r1r2x);
        __m256 r1r2y_sq = _mm256_mul_ps(r1r2y, r1r2y);
        __m256 r1r2z_sq = _mm256_mul_ps(r1r2z, r1r2z);

        // magnitude & mag squared of crossproduct
        __m256 square = _mm256_add_ps(r1r2x_sq, _mm256_add_ps(r1r2y_sq, r1r2z_sq));
        __m256 r1 = _mm256_sqrt_ps(_mm256_fmadd_ps(r1x, r1x, _mm256_fmadd_ps(r1y, r1y, _mm256_mul_ps(r1z, r1z))));
        __m256 r2 = _mm256_sqrt_ps(_mm256_fmadd_ps(r2x, r2x, _mm256_fmadd_ps(r2y, r2y, _mm256_mul_ps(r2z, r2z))));

        // (r1 - r2) x [r1 / |r1| - r2 / |r2|]
        // = (r1 x r2) * (|r2| - |r1|) / (|r1| * |r2|)  (simplification)
        // so lets define the scalar "factor" = (|r2| - |r1|) / (|r1| * |r2|)
        __m256 factor = _mm256_div_ps(_mm256_sub_ps(r2, r1), _mm256_mul_ps(r1, r2));

        // induced velocity (assume unit strength vortex)
        // v = (r1 x r2)^2 * factor / (4 * pi * |(r1 x r2)|^2)
        // v = (r1 x r2)^2 * coeff
        __m256 coeff = _mm256_div_ps(factor, _mm256_mul_ps(pi4, square));

        // add the influence and blend with mask
        // the masks should be done independently for optimal ILP but if compiler smart he can do it
        __m256 mask = _mm256_cmp_ps(r1, threshold, _CMP_LT_OS);
        mask = _mm256_or_ps(mask, _mm256_cmp_ps(r2, threshold, _CMP_LT_OS));
        mask = _mm256_or_ps(mask, _mm256_cmp_ps(square, threshold, _CMP_LT_OS));

        inf_x = _mm256_add_ps(inf_x, _mm256_blendv_ps(_mm256_mul_ps(r1r2x_sq, coeff), zero, mask));
        inf_y = _mm256_sub_ps(inf_y, _mm256_blendv_ps(_mm256_mul_ps(r1r2y_sq, coeff), zero, mask)); // HERE IS SUB INSTEAD OF ADD !
        inf_z = _mm256_add_ps(inf_z, _mm256_blendv_ps(_mm256_mul_ps(r1r2z_sq, coeff), zero, mask));
    }
}

void Solver::compute_lhs() {
    SimpleTimer timer("LHS");
    Mesh& m = mesh;
    const u32 v_ns = m.ns + 1;

    for (u32 i = 0; i < m.nc - 1; i++) {
        for (u32 j = 0; j < m.ns; j++) {
            // in sequential, these indices can be optimised away by just incrementing in the loop
            const u32 ia = i * m.ns + j;
            const u32 v0 = i * v_ns + j;
            const u32 v1 = v0 + 1;
            const u32 v3 = (i + 1) * v_ns + j;
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

            for (u32 ia2 = 0; ia2 < m.nb_panels_wing(); ia2+=8) {
                if (ia == 0 && ia2 == 8) {
                    for (u32 k = 0; k < 8; k++) std::cout << lhs[k] << std::endl;
                }
                // loads (3 regs)
                __m256 colloc_x = _mm256_loadu_ps(&m.colloc.x[ia2]);
                __m256 colloc_y = _mm256_loadu_ps(&m.colloc.y[ia2]);
                __m256 colloc_z = _mm256_loadu_ps(&m.colloc.z[ia2]);
                // 3 regs to store induced velocity
                __m256 inf_x = _mm256_setzero_ps();
                __m256 inf_y = _mm256_setzero_ps();
                __m256 inf_z = _mm256_setzero_ps();

                influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v0x, v0y, v0z, v1x, v1y, v1z);
                influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v1x, v1y, v1z, v2x, v2y, v2z);
                influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v2x, v2y, v2z, v3x, v3y, v3z);
                influence_avx2(inf_x, inf_y, inf_z, colloc_x, colloc_y, colloc_z, v3x, v3y, v3z, v0x, v0y, v0z);

                // dot product
                __m256 nx = _mm256_loadu_ps(&m.normal.x[ia2]);
                __m256 ny = _mm256_loadu_ps(&m.normal.y[ia2]);
                __m256 nz = _mm256_loadu_ps(&m.normal.z[ia2]);
                __m256 ring_inf = _mm256_fmadd_ps(inf_x, nx, _mm256_fmadd_ps(inf_y, ny, _mm256_mul_ps(inf_z, nz)));

                // store in col major order
                _mm256_storeu_ps(&lhs[ia * m.nb_panels_wing() + ia2], ring_inf);
            }
        }
    }

    for (u32 i = m.nc - 1; i < m.nc + m.nw; i++) {
        for (u32 j = 0; j < m.ns; j++) {
            const u32 ia = (m.nc - 1) * m.ns + j;
            const u32 v0 = i * v_ns + j;
            const u32 v1 = v0 + 1;
            const u32 v3 = (i + 1) * v_ns + j;
            const u32 v2 = v3 + 1;
            // in reality only load v1 & v2 and reuse data from previous v1 & v2 to be new v0 & v3 respectively
            // 12 regs (6 loads + 6 reuse) -> these will get spilled once we get in the influence function
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
            for (u32 i2 = 0; i2 < m.nc; i2++) {
                for (u32 j2 = 0; j2 < m.ns; j2++) {
                    const u32 ia2 = i2 * m.ns + j2;
                    // loads (3 regs)
                    const f32 colloc_x = m.colloc.x[ia2];
                    const f32 colloc_y = m.colloc.y[ia2];
                    const f32 colloc_z = m.colloc.z[ia2];

                    // 3 regs to store induced velocity 
                    Vec3 inf;
                    influence(inf, colloc_x, colloc_y, colloc_z, v0x, v0y, v0z, v1x, v1y, v1z);
                    influence(inf, colloc_x, colloc_y, colloc_z, v1x, v1y, v1z, v2x, v2y, v2z);
                    influence(inf, colloc_x, colloc_y, colloc_z, v2x, v2y, v2z, v3x, v3y, v3z);
                    influence(inf, colloc_x, colloc_y, colloc_z, v3x, v3y, v3z, v0x, v0y, v0z);

                    // storing in col major order
                    lhs[ia * m.nb_panels_wing() + ia2] += inf.x * m.normal.x[ia2] + inf.y * m.normal.y[ia2] + inf.z * m.normal.z[ia2];
                }
            }
        }
    }
}

void Solver::compute_rhs() {
    SimpleTimer timer("RHS");
    Data& d = data;
    Mesh& m = mesh;
    for (u32 i = 0; i < mesh.nb_panels_wing(); i++) {
        rhs[i] = - (d.u_inf.x * m.normal.x[i] + d.u_inf.y * m.normal.y[i] + d.u_inf.z * m.normal.z[i]);
    }
}

void Solver::solve() {
    SimpleTimer timer("Solve");
    const u32 n = mesh.nb_panels_wing();
    Eigen::Map<Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> A(lhs.data(), n, n);
    Eigen::Map<Eigen::VectorXf> x(data.gamma.data(), n);
    Eigen::Map<Eigen::VectorXf> b(rhs.data(), n);

    // Eigen::BiCGSTAB<Eigen::MatrixXf> solver;
    // solver.setTolerance(1.0e-5f);
    // solver.setMaxIterations(100);
    // solver.compute(A);
    // if (solver.info() != Eigen::Success) throw std::runtime_error("Failed to compute lhs matrix");
    // x = solver.solve(b);
    // if (solver.info() != Eigen::Success) throw std::runtime_error("Failed to converge");

    Eigen::PartialPivLU<Eigen::MatrixXf> lu(A);
    x = lu.solve(b);
}

void Solver::compute_forces() {
    Mesh& m = mesh;
    SimpleTimer timer("Compute forces");
    for (u32 i = 0; i < mesh.nb_panels_wing(); i++) {
        // force = U_inf x (panel_dy * rho * delta_gamma)
        // crossproduct:
        // (v0y*v1z - v0z*v1y);
        // (v0z*v1x - v0x*v1z);
        // (v0x*v1y - v0y*v1x);
        const f32 dl_x = (mesh.v1.x[i] - mesh.v0.x[i]) * data.rho * data.delta_gamma[i];
        const f32 dl_y = (mesh.v1.y[i] - mesh.v0.y[i]) * data.rho * data.delta_gamma[i];
        const f32 dl_z = (mesh.v1.z[i] - mesh.v0.z[i]) * data.rho * data.delta_gamma[i];
        // distance to reference calculated from the panel's force acting point (note: maybe precompute this?)
        const f32 dst_to_ref_x = data.ref_pt.x - 0.5f * (mesh.v0.x[i] + mesh.v1.x[i]);
        const f32 dst_to_ref_z = data.ref_pt.z - 0.5f * (mesh.v0.z[i] + mesh.v1.z[i]);

        const f32 force_x = data.u_inf.y * dl_z - data.u_inf.z * dl_y;
        const f32 force_y = data.u_inf.z * dl_x - data.u_inf.x * dl_z;
        const f32 force_z = data.u_inf.x * dl_y - data.u_inf.y * dl_x;
        
        // cl += force . lift_axis
        data.cl += force_x * data.lift_axis.x + force_y * data.lift_axis.y + force_z * data.lift_axis.z;
        // cm += force x distance_to_ref (only y component)
        data.cm += force_z * dst_to_ref_x - force_x * dst_to_ref_z;
    }

    for (u32 ia = mesh.nb_panels_wing(); ia < mesh.nb_panels_total(); ia++) {
        const f32 colloc_x = mesh.colloc.x[ia];
        const f32 colloc_y = mesh.colloc.y[ia];
        const f32 colloc_z = mesh.colloc.z[ia];
        Vec3 inf;
        for (u32 ia2 = mesh.nb_panels_wing(); ia2 < mesh.nb_panels_total(); ia2++) {
            Vec3 inf2;
            // Influence from the streamwise vortex lines
            influence(inf2, colloc_x, colloc_y, colloc_z, m.v1.x[ia2], m.v1.y[ia2], m.v1.z[ia2], m.v2.x[ia2], m.v2.y[ia2], m.v2.z[ia2]);
            influence(inf2, colloc_x, colloc_y, colloc_z, m.v3.x[ia2], m.v3.y[ia2], m.v3.z[ia2], m.v0.x[ia2], m.v0.y[ia2], m.v0.z[ia2]);
            f32 gamma_w = data.gamma[(m.nc-1)*m.ns + ia2 % m.ns];
            // This is the induced velocity calculated with the vortex (gamma) calculated earlier (according to kutta condition)
            inf.x += gamma_w * inf2.x;
            inf.y += gamma_w * inf2.y;
            inf.z += gamma_w * inf2.z;
        }
        const f32 w_ind = inf.x * m.normal.x[ia] + inf.y * m.normal.y[ia] + inf.z * m.normal.z[ia];
        const u32 col = ia % m.ns;
        const f32 dl = std::sqrt(pow<2>(mesh.v1.x[ia] - mesh.v0.x[ia]) + pow<2>(mesh.v1.y[ia] - mesh.v0.y[ia]) + pow<2>(mesh.v1.z[ia] - mesh.v0.z[ia]));
        
        data.cd -= 0.5f * data.rho * data.gamma[(m.nc-1)*m.ns + col] * w_ind * dl;
    }

    // this is = 1.0
    const f32 u_ref_mag_sqrd = pow<2>(data.u_inf.x) + pow<2>(data.u_inf.y) + pow<2>(data.u_inf.z);
    data.cl /= 0.5f * data.rho * u_ref_mag_sqrd * data.s_ref;
    data.cm /= 0.5f * data.rho * u_ref_mag_sqrd * data.s_ref * mesh.chord_avg();
    data.cd /= 0.5f * data.rho * u_ref_mag_sqrd * data.s_ref;

    // Cd = Cl^2 / (pi * AR * e) with AR = b^2 / S_ref
    // std::cout << "CD analytic: " << 2.0f * (data.cl * data.cl) / (PI_f * (10.0f*10.0f / data.s_ref)) << std::endl;
}
#include "vlm_backend_cpu_kernels.hpp"
#include "linalg.h"
#include "vlm_types.hpp"
#include <limits> 

using namespace vlm;
using namespace linalg::alias;

inline float3 kernel_biosavart(const float3& colloc, const float3& vertex1, const float3& vertex2, const f32& sigma) {
    const f32 rcut = 1e-10f; // highly tuned value
    const float3 r0 = vertex2 - vertex1;
    const float3 r1 = colloc - vertex1;
    const float3 r2 = colloc - vertex2;
    // Katz Plotkin, Low speed Aero | Eq 10.115
    const float3 r1r2cross = linalg::cross(r1, r2);
    const f32 r1_norm = linalg::length(r1);
    const f32 r2_norm = linalg::length(r2);
    const f32 square = linalg::length2(r1r2cross);
    if ((r1_norm<rcut) || (r2_norm<rcut) || (square<rcut)) {
        return float3{0.0f, 0.0f, 0.0f};
    }
    
    const f32 smoother = sigma*sigma*linalg::length2(r0);
    const f32 coeff = (linalg::dot(r0,r1)/r1_norm - linalg::dot(r0,r2)/r2_norm) / (4.0f*PI_f*std::sqrt(square*square + smoother*smoother));
    return r1r2cross * coeff;
}

inline void kernel_symmetry(float3& inf, float3 colloc, const float3& vertex0, const float3& vertex1, const f32 sigma) {
    float3 induced_speed = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf += induced_speed;
    colloc.y = -colloc.y; // wing symmetry
    float3 induced_speed_sym = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf.x += induced_speed_sym.x;
    inf.y -= induced_speed_sym.y;
    inf.z += induced_speed_sym.z;
}

void vlm::kernel_influence(
    u64 m, u64 n,
    f32 lhs[],
    f32 vx[], f32 vy[], f32 vz[],
    f32 collocx[], f32 collocy[], f32 collocz[],
    f32 normalx[], f32 normaly[], f32 normalz[],
    u64 ia, u64 lidx, f32 sigma
    ) {
    const u64 nb_panels = m * n;
    const u64 v0 = lidx + lidx / n;
    const u64 v1 = v0 + 1;
    const u64 v3 = v0 + n+1;
    const u64 v2 = v3 + 1;

    float3 vertex0{vx[v0], vy[v0], vz[v0]};
    float3 vertex1{vx[v1], vy[v1], vz[v1]};
    float3 vertex2{vx[v2], vy[v2], vz[v2]};
    float3 vertex3{vx[v3], vy[v3], vz[v3]};

    for (u64 ia2 = 0; ia2 < nb_panels; ia2++) {
        const float3 colloc(collocx[ia2], collocy[ia2], collocz[ia2]);
        float3 inf(0.0f, 0.0f, 0.0f);
        const float3 normal(normalx[ia2], normaly[ia2], normalz[ia2]);

        kernel_symmetry(inf, colloc, vertex0, vertex1, sigma);
        kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
        kernel_symmetry(inf, colloc, vertex2, vertex3, sigma);
        kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
        // store in col major order
        lhs[ia * nb_panels + ia2] += linalg::dot(inf, normal);
    }
}
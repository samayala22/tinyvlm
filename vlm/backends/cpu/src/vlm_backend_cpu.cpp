#include "vlm_backend_cpu.hpp"
#include "vlm_backend_cpu_kernels.hpp"
#include "vlm_backend_cpu_kernels_ispc.h"

#include "linalg.h"
#include "tinytimer.hpp"
#include "vlm_mesh.hpp"
#include "vlm_data.hpp"
#include "vlm_utils.hpp"
#include "vlm_executor.hpp" // includes taskflow/taskflow.hpp

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <immintrin.h>

#include <limits>
#include <taskflow/algorithm/for_each.hpp>

#include <lapacke.h>
#include <cblas.h>

using namespace vlm;

BackendCPU::~BackendCPU() = default; // Destructor definition

BackendCPU::BackendCPU(Mesh& mesh) : Backend(mesh) {
    lhs.resize((u64)mesh.nb_panels_wing() * (u64)mesh.nb_panels_wing());
    rhs.resize(mesh.nb_panels_wing());
    ipiv.resize(mesh.nb_panels_wing());
    gamma.resize(mesh.nb_panels_wing());
    delta_gamma.resize(mesh.nb_panels_wing());
    trefftz_buffer.resize(mesh.ns);
}

void BackendCPU::reset() {
    std::fill(gamma.begin(), gamma.end(), 0.0f);
    std::fill(lhs.begin(), lhs.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);
}

void BackendCPU::compute_delta_gamma() {
    std::copy(gamma.begin(), gamma.begin()+mesh.ns, delta_gamma.begin());

    // note: this is efficient as the memory is contiguous
    for (u32 i = 1; i < mesh.nc; i++) {
        for (u32 j = 0; j < mesh.ns; j++) {
            delta_gamma[i*mesh.ns + j] = gamma[i*mesh.ns + j] - gamma[(i-1)*mesh.ns + j];
        }
    }
}

void BackendCPU::compute_lhs(const FlowData& flow) {
    tiny::ScopedTimer timer("LHS");
    Mesh& m = mesh;

    ispc::MeshProxy mesh_proxy = {
        m.ns, m.nc, m.nb_panels_wing(),
        {m.v.x.data(), m.v.y.data(), m.v.z.data()}, 
        {m.colloc.x.data(), m.colloc.y.data(), m.colloc.z.data()},
        {m.normal.x.data(), m.normal.y.data(), m.normal.z.data()}
    };

    const u32 start_wing = 0;
    const u32 end_wing = (m.nc - 1) * m.ns;

    tf::Taskflow taskflow;

    auto init = taskflow.placeholder();
    auto wing_pass = taskflow.for_each_index(start_wing, end_wing, [&] (u32 i) {
        ispc::kernel_influence(mesh_proxy, lhs.data(), i, i, flow.sigma_vatistas);
    });

    u32 idx = m.nc - 1;
    auto cond = taskflow.emplace([&]{
        return idx < m.nc + m.nw ? 0 : 1; // 0 means continue, 1 means break
    });
    auto wake_pass = taskflow.for_each_index(0u, m.ns, [&] (u32 j) {
        const u32 ia = (m.nc - 1) * m.ns + j;
        const u32 lidx = idx * m.ns + j;
        ispc::kernel_influence(mesh_proxy, lhs.data(), ia, lidx, flow.sigma_vatistas);
    });
    auto back = taskflow.emplace([&]{
        idx++;
        return 0; // 0 means continue
    });
    auto sync = taskflow.placeholder();

    init.precede(wing_pass, cond);
    wing_pass.precede(sync);
    cond.precede(wake_pass, sync);
    wake_pass.precede(back);
    back.precede(cond);

    Executor::get().run(taskflow).wait();
}

void kernel_cpu_rhs(u32 n, const float normal_x[], const float normal_y[], const float normal_z[], float freestream_x, float freestream_y, float freestream_z, float rhs[]) {
    for (u32 i = 0; i < n; i++) {
        rhs[i] = - (freestream_x * normal_x[i] + freestream_y * normal_y[i] + freestream_z * normal_z[i]);
    }
}

void BackendCPU::compute_rhs(const FlowData& flow) {
    tiny::ScopedTimer timer("RHS");
    const Mesh& m = mesh;
    
    kernel_cpu_rhs(m.nb_panels_wing(), m.normal.x.data(), m.normal.y.data(), m.normal.z.data(), flow.freestream.x, flow.freestream.y, flow.freestream.z, rhs.data());
}

void BackendCPU::compute_rhs(const FlowData& flow, const std::vector<f32>& section_alphas) {
    tiny::ScopedTimer timer("Rebuild RHS");
    assert(section_alphas.size() == mesh.ns);
    const Mesh& m = mesh;
    for (u32 i = 0; i < mesh.nc; i++) {
        for (u32 j = 0; j < mesh.ns; j++) {
            const u32 li = i * mesh.ns + j; // linear index
            const linalg::alias::float3 freestream = compute_freestream(flow.u_inf, section_alphas[j], flow.beta);
            rhs[li] = - (freestream.x * m.normal.x[li] + freestream.y * m.normal.y[li] + freestream.z * m.normal.z[li]);
        }
    }
}

void BackendCPU::lu_factor() {
    tiny::ScopedTimer timer("Factor");
    const u32 n = mesh.nb_panels_wing();
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, lhs.data(), n, ipiv.data());
}

void BackendCPU::lu_solve() {
    tiny::ScopedTimer timer("Solve");
    const u32 n = mesh.nb_panels_wing();
    std::copy(rhs.begin(), rhs.end(), gamma.begin());

    LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, 1, lhs.data(), n, ipiv.data(), gamma.data(), n);
}

f32 BackendCPU::compute_coefficient_cl(const FlowData& flow, const f32 area,
    const u32 j, const u32 n) {
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    
    f32 cl = 0.0f;

    for (u32 u = 0; u < mesh.nc; u++) {
        for (u32 v = j; v < j + n; v++) {
            const u32 li = u * mesh.ns + v; // linear index
            const linalg::alias::float3 v0 = mesh.get_v0(li);
            const linalg::alias::float3 v1 = mesh.get_v1(li);
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = v1 - v0;
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * delta_gamma[li];
            cl += linalg::dot(force, flow.lift_axis);
        }
    }
    cl /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area;

    return cl;
}

linalg::alias::float3 BackendCPU::compute_coefficient_cm(
    const FlowData& flow,
    const f32 area,
    const f32 chord,
    const u32 j,
    const u32 n)
{
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    linalg::alias::float3 cm(0.f, 0.f, 0.f);

    for (u32 u = 0; u < mesh.nc; u++) {
        for (u32 v = j; v < j + n; v++) {
            const u32 li = u * mesh.ns + v; // linear index
            const linalg::alias::float3 v0 = mesh.get_v0(li);
            const linalg::alias::float3 v1 = mesh.get_v1(li);
            // Leading edge vector pointing outward from wing root
            const linalg::alias::float3 dl = v1 - v0;
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 dst_to_ref = mesh.ref_pt - 0.5f * (v0 + v1);
            // Distance from the center of leading edge to the reference point
            const linalg::alias::float3 force = linalg::cross(flow.freestream, dl) * flow.rho * delta_gamma[li];
            cm += linalg::cross(force, dst_to_ref);
        }
    }
    cm /= 0.5f * flow.rho * linalg::length2(flow.freestream) * area * chord;
    return cm;
}

inline void kernel_biosavart(f32& vx, f32& vy, f32& vz, f32& x, f32& y, f32& z, f32& x1, f32& y1, f32& z1, f32& x2, f32& y2, f32& z2) {
    static const f32 rcut = 1.0e-10f;
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

inline void kernel_symmetry(f32& inf_x, f32& inf_y, f32& inf_z, f32 x, f32 y, f32 z, f32 x1, f32 y1, f32 z1, f32 x2, f32 y2, f32 z2) {
    f32 vx, vy, vz;
    kernel_biosavart(vx, vy, vz, x, y, z, x1, y1, z1, x2, y2, z2);
    inf_x += vx;
    inf_y += vy;
    inf_z += vz;
    y = -y; // wing symmetry
    kernel_biosavart(vx, vy, vz, x, y, z, x1, y1, z1, x2, y2, z2);
    inf_x += vx;
    inf_y -= vy;
    inf_z += vz;
}

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

f32 BackendCPU::compute_coefficient_cd(
    const FlowData& flow,
    const f32 area,
    const u32 j,
    const u32 n) 
{
    tiny::ScopedTimer timer("Compute CD");
    assert(n > 0);
    assert(j >= 0 && j+n <= mesh.ns);
    std::fill(trefftz_buffer.begin(), trefftz_buffer.end(), 0.0f);
    
    f32 cd = 0.0f;
    // Drag coefficent computed using Trefftz plane
    const u32 begin = j + mesh.nb_panels_wing();
    const u32 end = begin + n;
    // parallel for
    for (u32 ia = begin; ia < end; ia++) {
        const u32 v0 = ia + ia / mesh.ns;
        const u32 v1 = v0 + 1;
        const u32 v3 = v0 + n+1;
        const u32 v2 = v3 + 1;

        const float3 vertex0{mesh.v.x[v0], mesh.v.y[v0], mesh.v.z[v0]};
        const float3 vertex1{mesh.v.x[v1], mesh.v.y[v1], mesh.v.z[v1]};
        const float3 vertex2{mesh.v.x[v2], mesh.v.y[v2], mesh.v.z[v2]};
        const float3 vertex3{mesh.v.x[v3], mesh.v.y[v3], mesh.v.z[v3]};

        const f32 gammaw = gamma[ia - mesh.ns];

        for (u32 ia2 = begin; ia2 < end; ia2++) {
            const float3 colloc(mesh.colloc.x[ia2], mesh.colloc.y[ia2], mesh.colloc.z[ia2]);
            const float3 normal(mesh.normal.x[ia2], mesh.normal.y[ia2], mesh.normal.z[ia2]);
            linalg::alias::float3 inf2(0.f, 0.f, 0.f);
            // Influence from the streamwise vortex lines
            kernel_symmetry(inf2, colloc, vertex1, vertex2, flow.sigma_vatistas);
            kernel_symmetry(inf2, colloc, vertex3, vertex0, flow.sigma_vatistas);
            // This is the induced velocity calculated with the vortex (gamma) calculated earlier (according to kutta condition)
            trefftz_buffer[ia2 - begin] += gammaw * linalg::dot(inf2, normal);
        }
    }

    for (u32 i = 0; i < mesh.ns; i++) {
        const u32 li = (mesh.nc-1) * mesh.ns + i;
        const f32 dl = mesh.strip_width(i);
        cd -= gamma[li] * trefftz_buffer[i] * dl; // used to have 0.5f * flow.rho
    }
    cd /= linalg::length2(flow.freestream) * area;
    return cd;
}
typedef float<3> float3;

#define DOT(v0, v1) (v0.x*v1.x + v0.y*v1.y + v0.z*v1.z)

#define CROSS(v0, v1, res) \
    res.x = v0.y*v1.z - v0.z*v1.y; \
    res.y = v0.z*v1.x - v0.x*v1.z; \
    res.z = v0.x*v1.y - v0.y*v1.x;

#define LENGTH2(v) (v.x*v.x + v.y*v.y + v.z*v.z)

#define LENGTH(v) (sqrt(LENGTH2(v)))

inline float dot(const float3& v0, const float3& v1) {return DOT(v0, v1);}
inline float dot(const float3& v0, const uniform float3& v1) {return DOT(v0, v1);}
inline float dot(const uniform float3& v0, const float3& v1) {return DOT(v0, v1);}
inline uniform float dot(const uniform float3& v0, const uniform float3& v1) {return DOT(v0, v1);}

inline float3 cross(const float3& v0, const float3& v1) {float3 res; CROSS(v0, v1, res); return res;}
inline float3 cross(const float3& v0, const uniform float3& v1) {float3 res; CROSS(v0, v1, res); return res;}
inline float3 cross(const uniform float3& v0, const float3& v1) {float3 res; CROSS(v0, v1, res); return res;}
inline uniform float3 cross(const uniform float3& v0, const uniform float3& v1) {uniform float3 res; CROSS(v0, v1, res); return res;}

inline float length2(const float3& v) {return LENGTH2(v);}
inline uniform float length2(const uniform float3& v) {return LENGTH2(v);}

inline float length(const float3& v) {return LENGTH(v);}
inline uniform float length(const uniform float3& v) {return LENGTH(v);}

inline uniform float3 normalize(const uniform float3& v) {
    uniform float l = length(v);
    const uniform float3 res = {v.x/l, v.y/l, v.z/l};
    return res;
}

inline float3 normalize(const float3& v) {
    float l = length(v);
    const varying float3 res = {v.x/l, v.y/l, v.z/l};
    return res;
}

template<typename F3>
inline F3 quad_normal(const F3& v0, const F3& v1, const F3& v2, const F3& v3) {
    return normalize(cross(v3-v1, v2-v0));
}

// Aggregate Structures
export struct SoA3D {
    uniform float* uniform x;
    uniform float* uniform y;
    uniform float* uniform z;
};

export struct MeshProxy {
    uniform uint64 ns;
    uniform uint64 nc;
    uniform uint64 nb_panels;
    uniform SoA3D v; // vertices
    uniform SoA3D colloc; // collocation points
    uniform SoA3D normal; // normals
};

export struct MeshView {
    uniform uint64 ni; // wing chord panels
    uniform uint64 nj; // wing span panels
    uniform uint64 nw; // wake chord capacity
    uniform uint64 cw; // number of active wake panels
    uniform SoA3D v; // vertices
    uniform SoA3D colloc; // collocation points
    uniform SoA3D normal; // normals
};

// Bio-savart Kernel
#define RCUT 1e-10f
#define RCUT2 1e-5f

#define PI_f 3.141593f

template<typename C, typename V>
inline float3 kernel_biosavart(C& colloc, const V& vertex1, const V& vertex2, const uniform float& sigma) {
    V r0 = vertex2 - vertex1;
    float3 r1 = colloc - vertex1;
    float3 r2 = colloc - vertex2;
    // Katz Plotkin, Low speed Aero | Eq 10.115
    float3 r1r2cross = cross(r1, r2);
    float r1_norm = length(r1);
    float r2_norm = length(r2);
    float square = length2(r1r2cross);
    
    if ((square<RCUT) || (r1_norm<RCUT2) || (r2_norm<RCUT2)) {
        float3 res = {0.0f, 0.0f, 0.0f};
        return res;
    }

    float smoother = sigma*sigma*length2(r0);

    float coeff = (dot(r0,r1)*r2_norm - dot(r0, r2)*r1_norm) / (4.0f*PI_f*sqrt(square*square + smoother*smoother)*r1_norm*r2_norm);
    return r1r2cross * coeff;
}

template<typename C, typename V>
inline void kernel_symmetry(float3& inf, C colloc, const V& vertex0, const V& vertex1, const uniform float& sigma) {
    float3 induced_speed = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf.x += induced_speed.x;
    inf.y += induced_speed.y;
    inf.z += induced_speed.z;
    colloc.y = -colloc.y; // wing symmetry
    float3 induced_speed_sym = kernel_biosavart(colloc, vertex0, vertex1, sigma);
    inf.x += induced_speed_sym.x;
    inf.y -= induced_speed_sym.y;
    inf.z += induced_speed_sym.z;
}

export void kernel_influence(
    uniform const MeshProxy& m,
    uniform float* uniform lhs,
    uniform uint64 ia, uniform uint64 lidx, uniform float sigma
    ) {

    const uniform uint64 v0 = lidx + lidx / m.ns;
    const uniform uint64 v1 = v0 + 1;
    const uniform uint64 v3 = v0 + m.ns+1;
    const uniform uint64 v2 = v3 + 1;


    // Broadcast vertices
    uniform float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
    uniform float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
    uniform float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
    uniform float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};

    // print("Influencer %: \n", lidx);
    // print("Vertex 0: % % %\n", vertex0.x, vertex0.y, vertex0.z);
    // print("Vertex 1: % % %\n", vertex1.x, vertex1.y, vertex1.z);
    // print("Vertex 2: % % %\n", vertex2.x, vertex2.y, vertex2.z);
    // print("Vertex 3: % % %\n", vertex3.x, vertex3.y, vertex3.z);

    foreach(ia2 = 0 ... m.nb_panels) {
        const float3 colloc = {m.colloc.x[ia2], m.colloc.y[ia2], m.colloc.z[ia2]};
        const float3 normal = {m.normal.x[ia2], m.normal.y[ia2], m.normal.z[ia2]};
        
        // print("Influenced: %\n", ia2);
        // print("Colloc: \n % \n % \n %\n", colloc.x, colloc.y, colloc.z);
        // print("Normal: \n % \n % \n %\n", normal.x, normal.y, normal.z);

        float3 inf = {0.0f, 0.0f, 0.0f};

        kernel_symmetry(inf, colloc, vertex0, vertex1, sigma);
        kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
        kernel_symmetry(inf, colloc, vertex2, vertex3, sigma);
        kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
        lhs[ia * m.nb_panels + ia2] += dot(inf, normal); // store
    }

    // for (uniform uint64 i = 0; i < m.nc; i++) {
    //     foreach(j = 0 ... m.ns) {
    //         uint64 ia2 = i * m.ns + j;

    //         uint64 vv0 = (i+0) * (m.ns+1) + j;
    //         uint64 vv1 = (i+0) * (m.ns+1) + j + 1;
    //         uint64 vv2 = (i+1) * (m.ns+1) + j + 1;
    //         uint64 vv3 = (i+1) * (m.ns+1) + j;

    //         // Loads
    //         const float3 vvertex0 = {m.v.x[vv0], m.v.y[vv0], m.v.z[vv0]};
    //         const float3 vvertex1 = {m.v.x[vv1], m.v.y[vv1], m.v.z[vv1]};
    //         const float3 vvertex2 = {m.v.x[vv2], m.v.y[vv2], m.v.z[vv2]};
    //         const float3 vvertex3 = {m.v.x[vv3], m.v.y[vv3], m.v.z[vv3]};

    //         const float3 colloc = 0.25f * (vvertex0 + vvertex1 + vvertex2 + vvertex3);
    //         //const float3 colloc = {m.colloc.x[ia2], m.colloc.y[ia2], m.colloc.z[ia2]};
    //         const float3 normal = quad_normal(vvertex0, vvertex1, vvertex2, vvertex3);

    //         float3 inf = {0.0f, 0.0f, 0.0f};
    //         kernel_symmetry(inf, colloc, vertex0, vertex1, sigma);
    //         kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
    //         kernel_symmetry(inf, colloc, vertex2, vertex3, sigma);
    //         kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
    //         lhs[ia * m.nb_panels + ia2] += dot(inf, normal); // store
    //     }
    // }
}

export void kernel_wake_influence(
    uniform MeshView& m,
    uniform uint64 lidx, uniform float* uniform gamma, uniform float* uniform rhs, uniform float sigma
    ) {

    const uniform float3 colloc_influenced = {m.colloc.x[lidx], m.colloc.y[lidx], m.colloc.z[lidx]};

    float3 induced_vel = {0.0f, 0.0f, 0.0f};

    // Wake influence on wing
    for (uint64 i = m.ni + m.nw - m.cw; i < m.ni + m.nw; i++) {
        foreach(j = 0 ... m.nj) {
            uint64 v0 = (i+0) * (m.nj+1) + j;
            uint64 v1 = (i+0) * (m.nj+1) + j + 1;
            uint64 v2 = (i+1) * (m.nj+1) + j + 1;
            uint64 v3 = (i+1) * (m.nj+1) + j;

            // Loads
            const float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
            const float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
            const float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
            const float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, colloc_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma[i * m.nj + j];
        }
    }

    const uniform float3 normal = {m.normal.x[lidx], m.normal.y[lidx], m.normal.z[lidx]};
    const float induced_vel_dot_normal = dot(induced_vel, normal); // vertical 2 fma + mul
    rhs[lidx] -= reduce_add(induced_vel_dot_normal);
}

export void kernel_rollup(
    uniform MeshView& m, uniform float dt, 
    uniform float* uniform rollup_vx, uniform float* uniform rollup_vy, uniform float* uniform rollup_vz,
    uniform uint64 vidx, uniform float* uniform gamma, uniform float sigma
    ) {

    const uniform float3 vertex_influenced = {m.v.x[vidx], m.v.y[vidx], m.v.z[vidx]};

    float3 induced_vel = {0.0f, 0.0f, 0.0f};

    // Wing influence
    for (uint64 i = 0; i < m.ni; i++) {
        foreach(j = 0 ... m.nj) {
            uint64 v0 = (i+0) * (m.nj+1) + j;
            uint64 v1 = (i+0) * (m.nj+1) + j + 1;
            uint64 v2 = (i+1) * (m.nj+1) + j + 1;
            uint64 v3 = (i+1) * (m.nj+1) + j;

            // Loads
            const float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
            const float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
            const float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
            const float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, vertex_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma[i * m.nj + j];
        }
    }

    // Wake influence
    for (uint64 i = m.ni + m.nw - m.cw; i < m.ni + m.nw; i++) {
        foreach(j = 0 ... m.nj) {
            uint64 v0 = (i+0) * (m.nj+1) + j;
            uint64 v1 = (i+0) * (m.nj+1) + j + 1;
            uint64 v2 = (i+1) * (m.nj+1) + j + 1;
            uint64 v3 = (i+1) * (m.nj+1) + j;

            // Loads
            const float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
            const float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
            const float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
            const float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, vertex_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma[i * m.nj + j];
        }
    }

    rollup_vx[vidx] = m.v.x[vidx] + reduce_add(induced_vel.x) * dt;
    rollup_vy[vidx] = m.v.y[vidx] + reduce_add(induced_vel.y) * dt;
    rollup_vz[vidx] = m.v.z[vidx] + reduce_add(induced_vel.z) * dt;
}

// export uniform float kernel_trefftz_cd(
//     uniform const MeshProxy& m,
//     uniform float* uniform gamma,
//     uniform float* uniform trefftz_buffer,
//     uniform uint64 j, uniform uint64 n, uniform float sigma
//     ) {
//     uniform uint64 begin = m.nb_panels + j;
//     uniform uint64 end = begin + n;
//     float cd = 0.0f;
    
//     // Compute the induced velocity of the streamwise wake vortex segments
//     for (uniform uint64 ia = m.nb_panels; ia < m.nb_panels + m.ns; ia++) {
//         const uniform uint64 v0 = ia + ia / m.ns;
//         const uniform uint64 v1 = v0 + 1;
//         const uniform uint64 v3 = v0 + m.ns+1;
//         const uniform uint64 v2 = v3 + 1;

//         // Broadcast vertices
//         const uniform float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
//         const uniform float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
//         const uniform float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
//         const uniform float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};

//         uniform float gammaw = gamma[ia - m.ns];
//         foreach(ia2 = begin ... end) {
//             const float3 colloc = {m.colloc.x[ia2], m.colloc.y[ia2], m.colloc.z[ia2]};
//             const float3 normal = {m.normal.x[ia2], m.normal.y[ia2], m.normal.z[ia2]};
//             float3 inf = {0.0f, 0.0f, 0.0f};

//             kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
//             kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
//             trefftz_buffer[ia2 - begin + j] += gammaw * dot(inf, normal); // store
//         }
//     }
//     // Perform the integration (Katz Plotkin, Low speed Aero | Eq 8.146)
//     foreach(i = j ... j + n) {
//         uint64 li = (m.nc-1) * m.ns + i;
//         float3 v0 = {m.v.x[i], m.v.y[i], m.v.z[i]};
//         float3 v1 = {m.v.x[i+1], m.v.y[i+1], m.v.z[i+1]};
//         float dl = length(v1 - v0);
//         cd -= gamma[li] * trefftz_buffer[i] * dl; // used to have 0.5f * flow.rho
//     }
//     return reduce_add(cd);
// }

export uniform float kernel_trefftz_cd2(
    uniform const MeshProxy& m,
    uniform float* uniform gamma,
    uniform uint32 jp, uniform uint32 n, uniform float sigma
    ) {
    varying float cd = 0.0f;

    // Loop over first wake panel spanwise section
    const uniform uint32 i = m.nc;
    // parallel for
    for (uniform uint32 j = jp; j < jp+n; j++) {
        uniform uint32 v0 = (i+0) * (m.ns+1) + j;
        uniform uint32 v1 = (i+0) * (m.ns+1) + j + 1;
        uniform uint32 v2 = (i+1) * (m.ns+1) + j + 1;
        uniform uint32 v3 = (i+1) * (m.ns+1) + j;
 
        // Loads
        const uniform float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
        const uniform float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
        const uniform float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
        const uniform float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};

        const uniform float3 colloc = 0.25f * (vertex0 + vertex1 + vertex2 + vertex3); // 3*(3 add + 1 mul)

        varying float3 inf = {0.0f, 0.0f, 0.0f};

        foreach(jj = jp ... jp+n) {
            varying uint32 vv0 = (i+0) * (m.ns+1) + jj;
            varying uint32 vv1 = (i+0) * (m.ns+1) + jj + 1;
            varying uint32 vv2 = (i+1) * (m.ns+1) + jj + 1;
            varying uint32 vv3 = (i+1) * (m.ns+1) + jj;

            // Loads
            const varying float3 vvertex0 = {m.v.x[vv0], m.v.y[vv0], m.v.z[vv0]};
            const varying float3 vvertex1 = {m.v.x[vv1], m.v.y[vv1], m.v.z[vv1]};
            const varying float3 vvertex2 = {m.v.x[vv2], m.v.y[vv2], m.v.z[vv2]};
            const varying float3 vvertex3 = {m.v.x[vv3], m.v.y[vv3], m.v.z[vv3]};
            varying float3 inf2 = {0.0f, 0.0f, 0.0f};

            kernel_symmetry(inf2, colloc, vvertex1, vvertex2, sigma);
            kernel_symmetry(inf2, colloc, vvertex3, vvertex0, sigma);

            varying float gammaw = gamma[(m.nc-1)*m.ns + jj];
            inf += gammaw * inf2;
        }

        const uniform float3 normal = quad_normal(vertex0, vertex1, vertex2, vertex3);
        const varying float induced_vel = dot(inf, normal);
        cd -= gamma[(m.nc-1)*m.ns + j] * induced_vel * length(vertex1 - vertex0);
    }
    return reduce_add(cd); // hadd, hadd, extract
}

export uniform float kernel_trefftz_cl(
    uniform const MeshProxy& m,
    uniform float* uniform gamma,
    uniform uint64 j, uniform uint64 n
    ) {
    float cl = 0.0f;
    foreach(i = j ... j + n) {
        uint64 li = (m.nc-1) * m.ns + i;
        float3 v0 = {m.v.x[i], m.v.y[i], m.v.z[i]};
        float3 v1 = {m.v.x[i+1], m.v.y[i+1], m.v.z[i+1]};
        float dl = length(v1 - v0);
        cl += gamma[li]* dl; // used to have 0.5f * flow.rho
    }
    return reduce_add(cl);
}

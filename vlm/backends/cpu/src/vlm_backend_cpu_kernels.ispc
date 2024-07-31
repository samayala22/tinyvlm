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

typedef float f32;
typedef double f64;
typedef int32 i32;
typedef int64 i64;
typedef uint32 u32;
typedef uint64 u64;

export struct Mesh {
    float* vertices; // (nc+nw+1)*(ns+1)*3
    // TODO: evaluate if we really need those values or we can just compute on the fly in the kernels
    float* normals; // nc*ns*3
    float* colloc; // nc*ns*3
    float* area; // nc*ns

    uint64 nc; // number of wing panels chordwise
    uint64 ns; // number of wing panels spanwise
    uint64 nw; // number of wake panels chordwise
    uint64 nwa; // number of wake panels active chordwise

    float s_ref; // reference area
    float c_ref; // reference chord
    float frame[16];

    int8* name; // mesh name
    bool lifting;
};

// Bio-savart Kernel
#define RCUT 1e-10f
#define RCUT2 1e-5f

#define PI_f 3.141593f
#define PTR_MESH_V(m, i,j,k) (m->vertices + (j) + (i) * (m->ns+1) + (k) * (m->nc+m->nw+1) * (m->ns+1))
#define PTR_MESH_C(m, i,j,k) (m->colloc + (j) + (i) * (m->ns) + (k) * (m->nc) * (m->ns))
#define PTR_MESH_N(m, i,j,k) (m->normals + (j) + (i) * (m->ns) + (k) * (m->nc) * (m->ns))

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

// v_rld : influencer vertices row leading dimension = nb of points span wise
export void kernel_influence(uniform u64 m, uniform f32* uniform lhs, uniform f32* uniform collocs, uniform u64 collocs_ld, uniform f32* uniform v, uniform u64 v_ld, uniform u64 v_rld, uniform f32* uniform normals, uniform u64 normals_ld, uniform f32 sigma) {
    
    uniform float3 vertex0 = {v[0 * v_ld]           , v[1 * v_ld]           , v[2 * v_ld]           };
    uniform float3 vertex1 = {v[0 * v_ld + 1]       , v[1 * v_ld + 1]       , v[2 * v_ld + 1]       };
    uniform float3 vertex2 = {v[0 * v_ld + v_rld +1], v[1 * v_ld + v_rld +1], v[2 * v_ld + v_rld +1]};
    uniform float3 vertex3 = {v[0 * v_ld + v_rld]   , v[1 * v_ld + v_rld]   , v[2 * v_ld + v_rld]   };

    // print("Influencer %: \n", lidx);
    // print("Vertex 0: % % %\n", vertex0.x, vertex0.y, vertex0.z);
    // print("Vertex 1: % % %\n", vertex1.x, vertex1.y, vertex1.z);
    // print("Vertex 2: % % %\n", vertex2.x, vertex2.y, vertex2.z);
    // print("Vertex 3: % % %\n", vertex3.x, vertex3.y, vertex3.z);

    foreach(i = 0 ... m) {
        const float3 colloc = {collocs[0 * collocs_ld + i], collocs[1 * collocs_ld + i], collocs[2 * collocs_ld + i]};
        const float3 normal = {normals[0 * normals_ld + i], normals[1 * normals_ld + i], normals[2 * normals_ld + i]};
        
        // print("Influenced: %\n", ia2);
        // print("Colloc: \n % \n % \n %\n", colloc.x, colloc.y, colloc.z);
        // print("Normal: \n % \n % \n %\n", normal.x, normal.y, normal.z);

        float3 inf = {0.0f, 0.0f, 0.0f};

        kernel_symmetry(inf, colloc, vertex0, vertex1, sigma);
        kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
        kernel_symmetry(inf, colloc, vertex2, vertex3, sigma);
        kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
        lhs[i] += dot(inf, normal); // store
    }
}

export void kernel_wake_influence(
    uniform const Mesh* uniform m,
    uniform uint64 lidx, uniform float* uniform gamma, uniform float* uniform rhs, uniform float sigma
    ) {
    const uniform uint64 nb_panels = m->ns * m->nc;
    const uniform float3 colloc_influenced = {m->colloc[lidx + 0*nb_panels], m->colloc[lidx + 1*nb_panels], m->colloc[lidx + 2*nb_panels]};

    float3 induced_vel = {0.0f, 0.0f, 0.0f};

    const uniform float* uniform vx = PTR_MESH_V(m, 0, 0, 0);
    const uniform float* uniform vy = PTR_MESH_V(m, 0, 0, 1);
    const uniform float* uniform vz = PTR_MESH_V(m, 0, 0, 2);

    // Wake influence on wing
    for (uint64 i = m->nc + m->nw - m->nwa; i < m->nc + m->nw; i++) {
        foreach(j = 0 ... m->ns) {
            uint64 v0 = (i+0) * (m->ns+1) + j;
            uint64 v1 = (i+0) * (m->ns+1) + j + 1;
            uint64 v2 = (i+1) * (m->ns+1) + j + 1;
            uint64 v3 = (i+1) * (m->ns+1) + j;

            // Loads
            const float3 vertex0 = {vx[v0], vy[v0], vz[v0]};
            const float3 vertex1 = {vx[v1], vy[v1], vz[v1]};
            const float3 vertex2 = {vx[v2], vy[v2], vz[v2]};
            const float3 vertex3 = {vx[v3], vy[v3], vz[v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, colloc_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma[i * m->ns + j];
        }
    }

    const uniform float* uniform nx = PTR_MESH_N(m, 0, 0, 0);
    const uniform float* uniform ny = PTR_MESH_N(m, 0, 0, 1);
    const uniform float* uniform nz = PTR_MESH_N(m, 0, 0, 2);

    const uniform float3 normal = {nx[lidx], ny[lidx], nz[lidx]};
    const float induced_vel_dot_normal = dot(induced_vel, normal); // vertical 2 fma + mul
    rhs[lidx] -= reduce_add(induced_vel_dot_normal);
}

export void kernel_rollup(
    uniform const Mesh* uniform m, uniform float dt, 
    uniform float* uniform rollup,
    uniform uint64 vidx, uniform float* uniform gamma, uniform float sigma
    ) {
    const uniform uint64 nb_verts_total = (m->nc+m->nw+1) * (m->ns+1);

    const uniform float* uniform vx = PTR_MESH_V(m, 0, 0, 0);
    const uniform float* uniform vy = PTR_MESH_V(m, 0, 0, 1);
    const uniform float* uniform vz = PTR_MESH_V(m, 0, 0, 2);
    uniform float* uniform rollup_vx = rollup + 0 * nb_verts_total;
    uniform float* uniform rollup_vy = rollup + 1 * nb_verts_total;
    uniform float* uniform rollup_vz = rollup + 2 * nb_verts_total;
    const uniform float3 vertex_influenced = {vx[vidx], vy[vidx], vz[vidx]};

    float3 induced_vel = {0.0f, 0.0f, 0.0f};

    // Wing influence
    for (uint64 i = 0; i < m->nc; i++) {
        foreach(j = 0 ... m->ns) {
            uint64 v0 = (i+0) * (m->ns+1) + j;
            uint64 v1 = (i+0) * (m->ns+1) + j + 1;
            uint64 v2 = (i+1) * (m->ns+1) + j + 1;
            uint64 v3 = (i+1) * (m->ns+1) + j;

            // Loads
            const float3 vertex0 = {vx[v0], vy[v0], vz[v0]};
            const float3 vertex1 = {vx[v1], vy[v1], vz[v1]};
            const float3 vertex2 = {vx[v2], vy[v2], vz[v2]};
            const float3 vertex3 = {vx[v3], vy[v3], vz[v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, vertex_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma[i * m->ns + j];
        }
    }

    // Wake influence
    for (uint64 i = m->nc + m->nw - m->nwa; i < m->nc + m->nw; i++) {
        foreach(j = 0 ... m->ns) {
            uint64 v0 = (i+0) * (m->ns+1) + j;
            uint64 v1 = (i+0) * (m->ns+1) + j + 1;
            uint64 v2 = (i+1) * (m->ns+1) + j + 1;
            uint64 v3 = (i+1) * (m->ns+1) + j;

            // Loads
            const float3 vertex0 = {vx[v0], vy[v0], vz[v0]};
            const float3 vertex1 = {vx[v1], vy[v1], vz[v1]};
            const float3 vertex2 = {vx[v2], vy[v2], vz[v2]};
            const float3 vertex3 = {vx[v3], vy[v3], vz[v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, vertex_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma[i * m->ns + j];
        }
    }

    rollup_vx[vidx] = vx[vidx] + reduce_add(induced_vel.x) * dt;
    rollup_vy[vidx] = vy[vidx] + reduce_add(induced_vel.y) * dt;
    rollup_vz[vidx] = vz[vidx] + reduce_add(induced_vel.z) * dt;
}

// export uniform float kernel_trefftz_cd(
//     uniform const Mesh* uniform m,
//     uniform float* uniform gamma,
//     uniform float* uniform trefftz_buffer,
//     uniform uint64 j, uniform uint64 n, uniform float sigma
//     ) {
//     const uniform uint64 nb_panels = m->ns * m->nc;
//     uniform uint64 begin = nb_panels + j;
//     uniform uint64 end = begin + n;
//     float cd = 0.0f;

//     const uniform float* uniform vx = PTR_MESH_V(m, 0, 0, 0);
//     const uniform float* uniform vy = PTR_MESH_V(m, 0, 0, 1);
//     const uniform float* uniform vz = PTR_MESH_V(m, 0, 0, 2);
//     const uniform float* uniform cx = PTR_MESH_C(m, 0, 0, 0);
//     const uniform float* uniform cy = PTR_MESH_C(m, 0, 0, 1);
//     const uniform float* uniform cz = PTR_MESH_C(m, 0, 0, 2);
//     const uniform float* uniform nx = PTR_MESH_N(m, 0, 0, 0);
//     const uniform float* uniform ny = PTR_MESH_N(m, 0, 0, 1);
//     const uniform float* uniform nz = PTR_MESH_N(m, 0, 0, 2);

//     // Compute the induced velocity of the streamwise wake vortex segments
//     for (uniform uint64 ia = nb_panels; ia < nb_panels + m->ns; ia++) {
//         const uniform uint64 v0 = ia + ia / m->ns;
//         const uniform uint64 v1 = v0 + 1;
//         const uniform uint64 v3 = v0 + m->ns+1;
//         const uniform uint64 v2 = v3 + 1;

//         // Broadcast vertices
//         const float3 vertex0 = {vx[v0], vy[v0], vz[v0]};
//         const float3 vertex1 = {vx[v1], vy[v1], vz[v1]};
//         const float3 vertex2 = {vx[v2], vy[v2], vz[v2]};
//         const float3 vertex3 = {vx[v3], vy[v3], vz[v3]};

//         uniform float gammaw = gamma[ia - m->ns];
//         foreach(ia2 = begin ... end) {
//             const float3 colloc = {cx[ia2], cy[ia2], cz[ia2]};
//             const float3 normal = {nx[ia2], ny[ia2], nz[ia2]};
//             float3 inf = {0.0f, 0.0f, 0.0f};

//             kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
//             kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
//             trefftz_buffer[ia2 - begin + j] += gammaw * dot(inf, normal); // store
//         }
//     }
//     // Perform the integration (Katz Plotkin, Low speed Aero | Eq 8.146)
//     foreach(i = j ... j + n) {
//         uint64 li = (m->nc-1) * m->ns + i;
//         float3 v0 = {vx[i], vy[i], vz[i]};
//         float3 v1 = {vx[i+1], vy[i+1], vz[i+1]};
//         float dl = length(v1 - v0);
//         cd -= gamma[li] * trefftz_buffer[i] * dl; // used to have 0.5f * flow.rho
//     }
//     return reduce_add(cd);
// }

export uniform float kernel_trefftz_cd2(
    uniform const Mesh* uniform m,
    uniform float* uniform gamma,
    uniform uint64 jp, uniform uint64 n, uniform float sigma
    ) {
    varying float cd = 0.0f;

    const uniform float* uniform vx = PTR_MESH_V(m, 0, 0, 0);
    const uniform float* uniform vy = PTR_MESH_V(m, 0, 0, 1);
    const uniform float* uniform vz = PTR_MESH_V(m, 0, 0, 2);

    // Loop over first wake panel spanwise section
    const uniform uint64 i = m->nc;
    // parallel for
    for (uniform uint64 j = jp; j < jp+n; j++) {
        uniform uint64 v0 = (i+0) * (m->ns+1) + j;
        uniform uint64 v1 = (i+0) * (m->ns+1) + j + 1;
        uniform uint64 v2 = (i+1) * (m->ns+1) + j + 1;
        uniform uint64 v3 = (i+1) * (m->ns+1) + j;
 
        // Loads
        const uniform float3 vertex0 = {vx[v0], vy[v0], vz[v0]};
        const uniform float3 vertex1 = {vx[v1], vy[v1], vz[v1]};
        const uniform float3 vertex2 = {vx[v2], vy[v2], vz[v2]};
        const uniform float3 vertex3 = {vx[v3], vy[v3], vz[v3]};

        const uniform float3 colloc = 0.25f * (vertex0 + vertex1 + vertex2 + vertex3); // 3*(3 add + 1 mul)

        varying float3 inf = {0.0f, 0.0f, 0.0f};

        foreach(jj = jp ... jp+n) {
            varying uint64 vv0 = (i+0) * (m->ns+1) + jj;
            varying uint64 vv1 = (i+0) * (m->ns+1) + jj + 1;
            varying uint64 vv2 = (i+1) * (m->ns+1) + jj + 1;
            varying uint64 vv3 = (i+1) * (m->ns+1) + jj;

            // Loads
            const varying float3 vvertex0 = {vx[vv0], vy[vv0], vz[vv0]};
            const varying float3 vvertex1 = {vx[vv1], vy[vv1], vz[vv1]};
            const varying float3 vvertex2 = {vx[vv2], vy[vv2], vz[vv2]};
            const varying float3 vvertex3 = {vx[vv3], vy[vv3], vz[vv3]};
            varying float3 inf2 = {0.0f, 0.0f, 0.0f};

            kernel_symmetry(inf2, colloc, vvertex1, vvertex2, sigma);
            kernel_symmetry(inf2, colloc, vvertex3, vvertex0, sigma);

            varying float gammaw = gamma[(m->nc-1)*m->ns + jj];
            inf += gammaw * inf2;
        }

        const uniform float3 normal = quad_normal(vertex0, vertex1, vertex2, vertex3);
        const varying float induced_vel = dot(inf, normal);
        cd -= gamma[(m->nc-1)*m->ns + j] * induced_vel * length(vertex1 - vertex0);
    }
    return reduce_add(cd); // hadd, hadd, extract
}

export uniform float kernel_trefftz_cl(
    uniform const Mesh* uniform m,
    uniform float* uniform gamma,
    uniform uint64 j, uniform uint64 n
    ) {
    float cl = 0.0f;
    const uniform float* uniform vx = PTR_MESH_V(m, 0, 0, 0);
    const uniform float* uniform vy = PTR_MESH_V(m, 0, 0, 1);
    const uniform float* uniform vz = PTR_MESH_V(m, 0, 0, 2);
    foreach(i = j ... j + n) {
        uint64 li = (m->nc-1) * m->ns + i;
        float3 v0 = {vx[i], vy[i], vz[i]};
        float3 v1 = {vx[i+1], vy[i+1], vz[i+1]};
        float dl = length(v1 - v0);
        cl += gamma[li]* dl; // used to have 0.5f * flow.rho
    }
    return reduce_add(cl);
}

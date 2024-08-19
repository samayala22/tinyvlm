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

// v_rld : influencer vertices row leading dimension = nb of points span wise
// Influence of a wing vortex ring (vertex quad) on all the collocation points of all wings
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

/// @brief Influence of a specific wing wake on a collocation point
/// @param colloc ptr to x pos of influenced collocation point
/// @param colloc_ld leading dimension of colloc
/// @param normals ptr to x pos of normals
/// @param normals_ld leading dimension of normals
/// @param v ptr to beginning of wake vertices
/// @param v_ld leading dimension of v (for each 3D dimension)
/// @param v_m number of wake vertices chordwise
/// @param v_n number of wake vertices spanwise
/// @param gamma ptr to begin of wake gamma
/// @param rhs ptr to rhs  value of influenced panel
/// @param sigma vortex core model factor
/// @param iteration current iteration number
export void kernel_wake_influence(uniform f32* uniform colloc, uniform u64 colloc_ld, uniform f32* uniform normals, uniform u64 normals_ld, uniform f32* uniform v, uniform u64 v_ld, uniform u64 v_m, uniform u64 v_n, uniform f32* uniform gamma, uniform f32* uniform rhs, uniform f32 sigma, uniform u32 iteration) {
    const uniform float3 colloc_influenced = {colloc[0*colloc_ld], colloc[1*colloc_ld], colloc[2*colloc_ld]};

    // print("Colloc: % % %\n", colloc_influenced.x, colloc_influenced.y, colloc_influenced.z);

    float3 induced_vel = {0.0f, 0.0f, 0.0f};

    // Wake influence on wing
    for (u64 i = v_m - iteration - 1; i < v_m-1; i++) {
        foreach(j = 0 ... v_n - 1) {
            const u64 v0 = (i+0) * v_n + j;
            const u64 v1 = (i+0) * v_n + j + 1;
            const u64 v2 = (i+1) * v_n + j + 1;
            const u64 v3 = (i+1) * v_n + j;

            // Loads
            const float3 vertex0 = {v[0*v_ld + v0], v[1*v_ld + v0], v[2*v_ld + v0]};
            const float3 vertex1 = {v[0*v_ld + v1], v[1*v_ld + v1], v[2*v_ld + v1]};
            const float3 vertex2 = {v[0*v_ld + v2], v[1*v_ld + v2], v[2*v_ld + v2]};
            const float3 vertex3 = {v[0*v_ld + v3], v[1*v_ld + v3], v[2*v_ld + v3]};

            // print("Vertex 0: % % %\n", vertex0.x, vertex0.y, vertex0.z);
            // print("Vertex 1: % % %\n", vertex1.x, vertex1.y, vertex1.z);
            // print("Vertex 2: % % %\n", vertex2.x, vertex2.y, vertex2.z);
            // print("Vertex 3: % % %\n", vertex3.x, vertex3.y, vertex3.z);

            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, colloc_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, colloc_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma[i * (v_n-1) + j];
        }
    }

    const uniform float3 normal = {normals[0*normals_ld], normals[1*normals_ld], normals[2*normals_ld]};
    const float induced_vel_dot_normal = dot(induced_vel, normal); // vertical 2 fma + mul
    *rhs -= reduce_add(induced_vel_dot_normal);
}

/// @brief Compute the new position of the wake vertices due to rollup effect (influence wing and wake vortex rings)
/// @param v_wake vertices of the wake vortex vertices (ptr to start of wing wake)
/// @param v_wake_ld leading dimension (for x, y and z) of the vertices of the wake vertices
/// @param v_wake_m number of wake vertices chord wise
/// @param v_wake_n number of wake vertices span wise
/// @param v_wake_idx index of the wake vertex to be updated
/// @param v_wing vertices of the wing vortex vertices (ptr to start of wing wake)
/// @param v_wing_ld leading dimension (for x, y and z) of the vertices of the wing vertices
/// @param v_wing_m number of wing vertices chord wise
/// @param v_wing_n number of wing vertices span wise
/// @param rollup vertices of the new wake vertices
/// @param rollup_ld leading dimension (for x, y and z) of the vertices of the new wake vertices
/// @param gamma_wing gamma vector of the wing vertices
/// @param gamma_wake gamma vector of the wake vertices
/// @param sigma vortex core model factor
/// @param dt time step
/// @param iteration current iteration number
export void kernel_rollup(uniform f32* uniform v_wake, uniform u64 v_wake_ld, uniform u64 v_wake_m, uniform u64 v_wake_n, uniform u64 v_wake_idx, uniform f32* uniform v_wing, uniform u64 v_wing_ld, uniform u64 v_wing_m, uniform u64 v_wing_n, uniform f32* uniform rollup, uniform u64 rollup_ld, uniform f32* uniform gamma_wing, uniform f32* uniform gamma_wake, uniform f32 sigma, uniform f32 dt, uniform u32 iteration) {
    const uniform float3 vertex_influenced = {v_wake[0*v_wake_ld + v_wake_idx], v_wake[1*v_wake_ld + v_wake_idx], v_wake[2*v_wake_ld + v_wake_idx]};

    float3 induced_vel = {0.0f, 0.0f, 0.0f}; 

    // Wing influence
    for (uint64 i = 0; i < v_wing_m - 1; i++) {
        foreach(j = 0 ... v_wing_n - 1) {
            u64 v0 = (i+0) * (v_wing_n) + j;
            u64 v1 = (i+0) * (v_wing_n) + j + 1;
            u64 v2 = (i+1) * (v_wing_n) + j + 1;
            u64 v3 = (i+1) * (v_wing_n) + j;

            // Loads
            const float3 vertex0 = {v_wing[0*v_wing_ld + v0], v_wing[1*v_wing_ld + v0], v_wing[2*v_wing_ld + v0]};
            const float3 vertex1 = {v_wing[0*v_wing_ld + v1], v_wing[1*v_wing_ld + v1], v_wing[2*v_wing_ld + v1]};
            const float3 vertex2 = {v_wing[0*v_wing_ld + v2], v_wing[1*v_wing_ld + v2], v_wing[2*v_wing_ld + v2]};
            const float3 vertex3 = {v_wing[0*v_wing_ld + v3], v_wing[1*v_wing_ld + v3], v_wing[2*v_wing_ld + v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, vertex_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma_wing[i * (v_wing_n-1) + j];
        }
    }

    // Wake influence
    for (u64 i = v_wake_m - iteration - 1; i < v_wake_m-1; i++) {
        foreach(j = 0 ... v_wake_n - 1) {
            u64 v0 = (i+0) * (v_wake_n) + j;
            u64 v1 = (i+0) * (v_wake_n) + j + 1;
            u64 v2 = (i+1) * (v_wake_n) + j + 1;
            u64 v3 = (i+1) * (v_wake_n) + j;

            // Loads
            const float3 vertex0 = {v_wake[0*v_wake_ld + v0], v_wake[1*v_wake_ld + v0], v_wake[2*v_wake_ld + v0]};
            const float3 vertex1 = {v_wake[0*v_wake_ld + v1], v_wake[1*v_wake_ld + v1], v_wake[2*v_wake_ld + v1]};
            const float3 vertex2 = {v_wake[0*v_wake_ld + v2], v_wake[1*v_wake_ld + v2], v_wake[2*v_wake_ld + v2]};
            const float3 vertex3 = {v_wake[0*v_wake_ld + v3], v_wake[1*v_wake_ld + v3], v_wake[2*v_wake_ld + v3]};
        
            float3 ind = {0.0f, 0.0f, 0.0f};
            kernel_symmetry(ind, vertex_influenced, vertex0, vertex1, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex1, vertex2, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex2, vertex3, sigma);
            kernel_symmetry(ind, vertex_influenced, vertex3, vertex0, sigma);

            induced_vel += ind * gamma_wake[i * (v_wake_n-1) + j];
        }
    }

    rollup[0*rollup_ld + v_wake_idx] = vertex_influenced.x + reduce_add(induced_vel.x) * dt;
    rollup[1*rollup_ld + v_wake_idx] = vertex_influenced.y + reduce_add(induced_vel.y) * dt;
    rollup[2*rollup_ld + v_wake_idx] = vertex_influenced.z + reduce_add(induced_vel.z) * dt;
}

export uniform float kernel_trefftz_cd(uniform f32* uniform v_wake, uniform u64 v_wake_ld, uniform u64 v_wake_m, uniform u64 v_wake_n, uniform float* uniform gamma_wake, uniform f32 sigma) {
    varying float cd = 0.0f;

    // Loop over first wake panel spanwise section
    const uniform u64 i = v_wake_m - 2; // 2nd to last wake chordwise index
    // parallel for
    for (uniform u64 j = 0; j < v_wake_n - 1; j++) {
        uniform u64 v0 = (i+0) * (v_wake_n) + j;
        uniform u64 v1 = (i+0) * (v_wake_n) + j + 1;
        uniform u64 v2 = (i+1) * (v_wake_n) + j + 1;
        uniform u64 v3 = (i+1) * (v_wake_n) + j;
 
        // Loads
        const uniform float3 vertex0 = {v_wake[0*v_wake_ld + v0], v_wake[1*v_wake_ld + v0], v_wake[2*v_wake_ld + v0]};
        const uniform float3 vertex1 = {v_wake[0*v_wake_ld + v1], v_wake[1*v_wake_ld + v1], v_wake[2*v_wake_ld + v1]};
        const uniform float3 vertex2 = {v_wake[0*v_wake_ld + v2], v_wake[1*v_wake_ld + v2], v_wake[2*v_wake_ld + v2]};
        const uniform float3 vertex3 = {v_wake[0*v_wake_ld + v3], v_wake[1*v_wake_ld + v3], v_wake[2*v_wake_ld + v3]};

        const uniform float3 colloc = 0.25f * (vertex0 + vertex1 + vertex2 + vertex3); // 3*(3 add + 1 mul)

        varying float3 inf = {0.0f, 0.0f, 0.0f};

        foreach(jj = 0 ... v_wake_n - 1) {
            varying u64 vv0 = (i+0) * (v_wake_n) + jj;
            varying u64 vv1 = (i+0) * (v_wake_n) + jj + 1;
            varying u64 vv2 = (i+1) * (v_wake_n) + jj + 1;
            varying u64 vv3 = (i+1) * (v_wake_n) + jj;

            // Loads
            const varying float3 vvertex0 = {v_wake[0*v_wake_ld + vv0], v_wake[1*v_wake_ld + vv0], v_wake[2*v_wake_ld + vv0]};
            const varying float3 vvertex1 = {v_wake[0*v_wake_ld + vv1], v_wake[1*v_wake_ld + vv1], v_wake[2*v_wake_ld + vv1]};
            const varying float3 vvertex2 = {v_wake[0*v_wake_ld + vv2], v_wake[1*v_wake_ld + vv2], v_wake[2*v_wake_ld + vv2]};
            const varying float3 vvertex3 = {v_wake[0*v_wake_ld + vv3], v_wake[1*v_wake_ld + vv3], v_wake[2*v_wake_ld + vv3]};
            varying float3 inf2 = {0.0f, 0.0f, 0.0f};

            kernel_symmetry(inf2, colloc, vvertex1, vvertex2, sigma);
            kernel_symmetry(inf2, colloc, vvertex3, vvertex0, sigma);

            varying float gammaw = gamma_wake[i * (v_wake_n-1) + jj];
            inf += gammaw * inf2;
        }

        const uniform float3 normal = quad_normal(vertex0, vertex1, vertex2, vertex3);
        const varying float induced_vel = dot(inf, normal);
        cd -= gamma_wake[i * (v_wake_n-1) + j] * induced_vel * length(vertex1 - vertex0);
    }
    return reduce_add(cd); // hadd, hadd, extract
}

// export uniform float kernel_trefftz_cl(
//     uniform const Mesh* uniform m,
//     uniform float* uniform gamma,
//     uniform uint64 j, uniform uint64 n
//     ) {
//     float cl = 0.0f;
//     const uniform float* uniform vx = PTR_MESH_V(m, 0, 0, 0);
//     const uniform float* uniform vy = PTR_MESH_V(m, 0, 0, 1);
//     const uniform float* uniform vz = PTR_MESH_V(m, 0, 0, 2);
//     foreach(i = j ... j + n) {
//         uint64 li = (m->nc-1) * m->ns + i;
//         float3 v0 = {vx[i], vy[i], vz[i]};
//         float3 v1 = {vx[i+1], vy[i+1], vz[i+1]};
//         float dl = length(v1 - v0);
//         cl += gamma[li]* dl; // used to have 0.5f * flow.rho
//     }
//     return reduce_add(cl);
// }

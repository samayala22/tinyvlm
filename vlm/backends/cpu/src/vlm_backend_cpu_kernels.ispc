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
}

export void kernel_influence2(
    uniform const MeshProxy& m,
    uniform float* uniform lhs, uniform float* uniform uw, uniform float* uniform vw, uniform float* uniform ww,
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
        uw[ia * m.nb_panels + ia2] += inf.x;
        vw[ia * m.nb_panels + ia2] += inf.y;
        ww[ia * m.nb_panels + ia2] += inf.z;
    }
}

export void kernel_induced_vel(
    uniform const MeshView& m, uniform uint64 vidx, uniform float dt,
    uniform float* uniform gamma, uniform float sigma
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

    m.v.x[vidx] += reduce_add(induced_vel.x) * dt;
    m.v.y[vidx] += reduce_add(induced_vel.y) * dt;
    m.v.z[vidx] += reduce_add(induced_vel.z) * dt;
}

export uniform float kernel_trefftz_cd(
    uniform const MeshProxy& m,
    uniform float* uniform gamma,
    uniform float* uniform trefftz_buffer,
    uniform uint64 j, uniform uint64 n, uniform float sigma
    ) {
    uniform uint64 begin = m.nb_panels + j;
    uniform uint64 end = begin + n;
    float cd = 0.0f;
    
    // Compute the induced velocity of the streamwise wake vortex segments
    for (uniform uint64 ia = m.nb_panels; ia < m.nb_panels + m.ns; ia++) {
        const uniform uint64 v0 = ia + ia / m.ns;
        const uniform uint64 v1 = v0 + 1;
        const uniform uint64 v3 = v0 + m.ns+1;
        const uniform uint64 v2 = v3 + 1;

        // Broadcast vertices
        const uniform float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
        const uniform float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
        const uniform float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
        const uniform float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};

        uniform float gammaw = gamma[ia - m.ns];
        foreach(ia2 = begin ... end) {
            const float3 colloc = {m.colloc.x[ia2], m.colloc.y[ia2], m.colloc.z[ia2]};
            const float3 normal = {m.normal.x[ia2], m.normal.y[ia2], m.normal.z[ia2]};
            float3 inf = {0.0f, 0.0f, 0.0f};

            kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
            kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
            trefftz_buffer[ia2 - begin + j] += gammaw * dot(inf, normal); // store
        }
    }
    // Perform the integration (Katz Plotkin, Low speed Aero | Eq 8.146)
    foreach(i = j ... j + n) {
        uint64 li = (m.nc-1) * m.ns + i;
        float3 v0 = {m.v.x[i], m.v.y[i], m.v.z[i]};
        float3 v1 = {m.v.x[i+1], m.v.y[i+1], m.v.z[i+1]};
        float dl = length(v1 - v0);
        cd -= gamma[li] * trefftz_buffer[i] * dl; // used to have 0.5f * flow.rho
    }
    return reduce_add(cd);
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

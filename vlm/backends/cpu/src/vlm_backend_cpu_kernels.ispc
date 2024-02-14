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
    uniform unsigned int ns;
    uniform unsigned int nc;
    uniform unsigned int nb_panels;
    uniform SoA3D v; // vertices
    uniform SoA3D colloc; // collocation points
    uniform SoA3D normal; // normals
};

// Bio-savart Kernel
#define RCUT 1e-10f
#define RCUT2 1e-5f

#define PI_f 3.141593f

inline float3 kernel_biosavart(float3& colloc, const uniform float3& vertex1, const uniform float3& vertex2, const uniform float& sigma) {
    uniform float3 r0 = vertex2 - vertex1;
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

    uniform float smoother = sigma*sigma*length2(r0);

    float coeff = (dot(r0,r1)*r2_norm - dot(r0, r2)*r1_norm) / (4.0f*PI_f*sqrt(square*square + smoother*smoother)*r1_norm*r2_norm);
    return r1r2cross * coeff;
}

inline void kernel_symmetry(float3& inf, float3 colloc, const uniform float3& vertex0, const uniform float3& vertex1, const uniform float& sigma) {
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
    uniform unsigned int ia, uniform unsigned int lidx, uniform float sigma
    ) {
    const uniform unsigned int v0 = lidx + lidx / m.ns;
    const uniform unsigned int v1 = v0 + 1;
    const uniform unsigned int v3 = v0 + m.ns+1;
    const uniform unsigned int v2 = v3 + 1;

    // Broadcast vertices
    uniform float3 vertex0 = {m.v.x[v0], m.v.y[v0], m.v.z[v0]};
    uniform float3 vertex1 = {m.v.x[v1], m.v.y[v1], m.v.z[v1]};
    uniform float3 vertex2 = {m.v.x[v2], m.v.y[v2], m.v.z[v2]};
    uniform float3 vertex3 = {m.v.x[v3], m.v.y[v3], m.v.z[v3]};

    foreach(ia2 = 0 ... m.nb_panels) {
        const float3 colloc = {m.colloc.x[ia2], m.colloc.y[ia2], m.colloc.z[ia2]};
        const float3 normal = {m.normal.x[ia2], m.normal.y[ia2], m.normal.z[ia2]};
        float3 inf = {0.0f, 0.0f, 0.0f};

        kernel_symmetry(inf, colloc, vertex0, vertex1, sigma);
        kernel_symmetry(inf, colloc, vertex1, vertex2, sigma);
        kernel_symmetry(inf, colloc, vertex2, vertex3, sigma);
        kernel_symmetry(inf, colloc, vertex3, vertex0, sigma);
        lhs[ia * m.nb_panels + ia2] += dot(inf, normal); // store
    }
}

export uniform float kernel_trefftz_cd(
    uniform const MeshProxy& m,
    uniform float* uniform gamma,
    uniform float* uniform trefftz_buffer,
    uniform unsigned int j, uniform unsigned int n, uniform float sigma
    ) {
    uniform unsigned int begin = m.nb_panels + j;
    uniform unsigned int end = begin + n;
    float cd = 0.0f;
    
    // Compute the induced velocity of the streamwise wake vortex segments
    for (uniform unsigned int ia = m.nb_panels; ia < m.nb_panels + m.ns; ia++) {
        const uniform unsigned int v0 = ia + ia / m.ns;
        const uniform unsigned int v1 = v0 + 1;
        const uniform unsigned int v3 = v0 + m.ns+1;
        const uniform unsigned int v2 = v3 + 1;

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
        unsigned int li = (m.nc-1) * m.ns + i;
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
    uniform unsigned int j, uniform unsigned int n
    ) {
    float cl = 0.0f;
    foreach(i = j ... j + n) {
        unsigned int li = (m.nc-1) * m.ns + i;
        float3 v0 = {m.v.x[i], m.v.y[i], m.v.z[i]};
        float3 v1 = {m.v.x[i+1], m.v.y[i+1], m.v.z[i+1]};
        float dl = length(v1 - v0);
        cl += gamma[li]* dl; // used to have 0.5f * flow.rho
    }
    return reduce_add(cl);
}

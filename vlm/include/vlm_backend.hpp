#pragma once

#include "vlm_fwd.hpp"
#include "vlm_types.hpp"

namespace vlm {

class Backend {
    public:
        Mesh& mesh;
        Data& data;

        Backend(Mesh& mesh, Data& data) : mesh(mesh), data(data) {};
        virtual void reset() = 0;
        virtual void compute_lhs() = 0;
        virtual void compute_rhs() = 0;
        virtual void solve() = 0;
        virtual void compute_forces() = 0;
        virtual void compute_delta_gamma() = 0;
        virtual ~Backend() = default;
};

}

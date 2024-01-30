#include "vlm_data.hpp"
#include "vlm_inline.hpp"

#include "Eigen/src/Core/Matrix.h"
#include <cmath>

using namespace vlm;

FlowData::FlowData(const f32 alpha_, const f32 beta_, const f32 u_inf_, const f32 rho_): 
    alpha(alpha_), beta(beta_), u_inf(u_inf_), rho(rho_),
    freestream(compute_freestream(u_inf, alpha, beta)),
    lift_axis(compute_lift_axis(freestream)),
    stream_axis(compute_stream_axis(freestream)) {
}
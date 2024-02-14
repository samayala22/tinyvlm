#include "vlm_types.hpp"

namespace vlm {
void kernel_influence(
    u32 m, u32 n,
    f32 lhs[],
    f32 vx[], f32 vy[], f32 vz[],
    f32 collocx[], f32 collocy[], f32 collocz[],
    f32 normalx[], f32 normaly[], f32 normalz[],
    u32 ia, u32 lidx, f32 sigma_p4
    );
}


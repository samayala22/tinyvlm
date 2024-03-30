#include <cstdint>
#include <cmath>

namespace tiny {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
 
using f32 = float;
using f64 = double;

static const f64 PI_d = 3.14159265358979;
static const f32 PI_f = 3.141593f;
static const f64 EPS_d = 2.22045e-16;
static const f32 EPS_f = 1.19209e-07f;
static const f64 EPS_sqrt_d = std::sqrt(EPS_d);
static const f32 EPS_sqrt_f = std::sqrt(EPS_f);
static const f64 EPS_cbrt_d = std::cbrt(EPS_d);
static const f32 EPS_cbrt_f = std::cbrt(EPS_f);

} // namespace tiny
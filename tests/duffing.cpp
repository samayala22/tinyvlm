// Temporary file for implementing a continuation method

#include "tinytimer.hpp"
#include "tinytest.hpp"

#include "vlm.hpp"
#include "vlm_backend.hpp"
#include "vlm_types.hpp"

using namespace vlm;

void create_idft_matrices(
    const TensorView2fH& idft, // u_freq -> u
    const TensorView2fH& iddft, // u_freq -> du / dt
    const TensorView2fH& idddft, // u_freq -> ddu / dt^2
    const f32 omega, // base frequency
    const i32 harmonics // number of harmonics for the truncated fourier series
) {
    const i32 unknowns = 2 * harmonics + 1;
    const f32 period = 2.0f * PI_f / omega;

    f32 sqrt_unknowns0 = 1.f / std::sqrt((f32)unknowns);
    f32 sqrt_unknowns = std::sqrt(2.f) * sqrt_unknowns0;

    for (i64 i = 0; i < unknowns; i++) {
        idft(i, 0) = sqrt_unknowns0;
        iddft(i, 0) = 0;
        idddft(i, 0) = 0;
    }

    for (i64 j = 1; j < unknowns; j+=2) {
        f32 k = (f32)(j + 1) * 0.5f;
        for (i64 i = 0; i < unknowns; i++) {
            const f32 tn = (f32)i / (f32)unknowns * period;
            const f32 s = std::sin(omega * tn * k) * sqrt_unknowns;
            const f32 c = std::cos(omega * tn * k) * sqrt_unknowns;
            const f32 omega_k = omega * k;
            idft(i, j) = c;
            idft(i, j + 1) = s;
            iddft(i, j) = - omega_k * s;
            iddft(i, j + 1) = omega_k * c;
            idddft(i, j) = - omega_k * omega_k * c;
            idddft(i, j + 1) = - omega_k * omega_k * s;
        }
    }
}
#include "vlm.hpp"
#include "vlm_kinematics.hpp"

#include "tinytest.hpp"

#include <cmath>
#include <cstdio>
#include <cassert>
#include <string>

using namespace vlm;

int main(int /*unused*/, char** /*unused*/) {
    const f32 u_inf = 1.0f;
    const f32 amplitude = 0.1f;
    const f32 omega = 0.5f;

    KinematicsTree<f32> kinematics;
    auto body_init = kinematics.add([=](const fwd::Float& t) { return translation_matrix<fwd::Float>({0.0f, 10.0f, 0.0f}); }); // initial position
    auto freestream = kinematics.add([=](const fwd::Float& t) { return translation_matrix<fwd::Float>({-u_inf*t, 0.0f, 0.0f}); })->after(body_init); // freestream
    // auto heave = kinematics.add([=](const fwd::Float& t) { return translation_matrix<fwd::Float>({0.0f, 0.0f, amplitude * fwd::sin(omega * t)}); })->after(freestream); // heave
    auto pitch = kinematics.add([=](const fwd::Float& t) { return rotation_matrix<fwd::Float>({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f * PI_f * fwd::sin(omega * t)); })->after(freestream); // pitch
    
    Assembly assembly(freestream);
    assembly.add("../../../../mesh/infinite_rectangular_2x2.x", pitch);
    
    auto wing_kinematics = assembly.surface_kinematics()[0];

    const f32 t = 1.0f;
    auto vel = wing_kinematics->linear_velocity(t, {1.0f, 0.0f, 0.0f}); // point velocity
    auto ang_vel = wing_kinematics->angular_velocity(t);
    std::printf("t: %f, vel: %f %f %f\n", t, vel.x, vel.y, vel.z);
    std::printf("t: %f, ang vel: %f %f %f\n", t, ang_vel.x, ang_vel.y, ang_vel.z);

    TINY_ASSERT_NEAR(ang_vel.x, 0.0f, 1e-6);
    TINY_ASSERT_NEAR(ang_vel.y, 0.5f * PI_f * omega * std::cos(omega * t), 1e-6);
    TINY_ASSERT_NEAR(ang_vel.z, 0.0f, 1e-6);
    return 0;
}
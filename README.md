[![Windows](https://github.com/samayala22/tinyvlm/actions/workflows/windows.yaml/badge.svg)](https://github.com/samayala22/tinyvlm/actions/workflows/windows.yaml)
[![Linux](https://github.com/samayala22/tinyvlm/actions/workflows/linux.yaml/badge.svg)](https://github.com/samayala22/tinyvlm/actions/workflows/linux.yaml)
<!-- [![MacOS](https://github.com/AER8875-2022/AeroFLEX/actions/workflows/macos.yaml/badge.svg)](https://github.com/AER8875-2022/AeroFLEX/actions/workflows/macos.yaml) -->
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=samayala22_tinyvlm&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=samayala22_tinyvlm)

# TinyVLM

### High Performance Potential Flow Solver for Aeroelastic Analysis
---

## Features:

- [X] VLM
- [X] NL-VLM
- [X] UVLM
- [X] 2DOF UVLM
- [X] 3DOF UVLM
- [X] HBVLM
- [X] 2DOF HBVLM
- [X] 3DOF HBVLM
- [X] Continuation Solver

## Backends

- [X] [ISPC](https://github.com/ispc/ispc) + [Taskflow](https://github.com/taskflow/taskflow)
- [X] CUDA

## Installation

[xmake](https://xmake.io) is used as build system. 
- For the CPU backend the ISPC compiler and a C++ compiler are required.
- For the CUDA backend the CUDA Toolkit is required. 

Key commands:

```bash
# Configure (first time)
xmake f -p [windows|linux] --build-cpu=y --build-cuda=n -m release

# Build all default targets
xmake

# Build a specific target
xmake build uvlm_2dof
xmake build libhbvlm

# Run all tests
xmake test

# Run a specific test
xmake test vlm_elliptic_coeffs

# Generate compile_commands.json (for IDE/clangd)
xmake project -k compile_commands
```

## Typical Workflows

### Steady aerodynamic analysis

```
1. Pick or generate a mesh file (mesh/*.x)
2. Construct a Backend (CPU or CUDA)
3. Create a VLM or NLVLM solver
4. Set FlowData (alpha, u_inf, rho)
5. Call solver.run()
6. Query coeff_cl(), coeff_cm() for lift and moment
```

### Time-domain flutter analysis (UVLM + 2DOF)

```
1. Load wing mesh → two MultiTensor3f surfaces if 2DOF
2. Build KinematicsTree with heave and pitch nodes
3. Instantiate NewmarkBeta with [M, C, K] matrices
4. Time loop:
   a. Kinematics.transform(t) → displace_wing()
   b. wake_shed() → grow trailing wake
   c. lhs_assemble(), rhs_assemble_*() → solve for γ
   d. forces_unsteady() → aerodynamic loads
   e. NewmarkBeta.step() → update structural DOFs
   f. Feed new DOFs back into KinematicNode lambdas
5. Plot heave/pitch amplitude vs. time → flutter detected when amplitude grows
```

### Limit-cycle oscillation analysis (HBVLM + continuation)

```
1. Build and compile libhbvlm or libhbvlm3 (xmake build libhbvlm)
2. In Python:
   a. from libhbvlm import HBVLM
   b. hb = HBVLM(); hb.init(harmonics=5)
   c. Define structural residual in dof2.py (freeplay or cubic stiffness)
   d. Use harmonic_balance.py to assemble full residual R(X, omega)
   e. Use continuation.py to sweep omega (airspeed)
   f. Detect Hopf bifurcation, continue onto LCO branch
3. Plot amplitude vs. airspeed with plotting.py → produces build/*.html
```
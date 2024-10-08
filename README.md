[![Windows](https://github.com/samayala22/tinyvlm/actions/workflows/windows.yaml/badge.svg)](https://github.com/samayala22/tinyvlm/actions/workflows/windows.yaml)
[![Linux](https://github.com/samayala22/tinyvlm/actions/workflows/linux.yaml/badge.svg)](https://github.com/samayala22/tinyvlm/actions/workflows/linux.yaml)
<!-- [![MacOS](https://github.com/AER8875-2022/AeroFLEX/actions/workflows/macos.yaml/badge.svg)](https://github.com/AER8875-2022/AeroFLEX/actions/workflows/macos.yaml) -->
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=samayala22_tinyvlm&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=samayala22_tinyvlm)

# TinyVLM

### High Performance Potential Flow Solver
---

## Features:

- [X] VLM
- [X] NL-VLM
- [X] UVLM
- [ ] NL-UVLM
- [ ] NLHB-UVLM

# Correctors

- [X] High angle of attack correction
- [X] Dihedral / Anhedral Wings (local coordinate projection in force calculation)
- [X] Swept Wings (coordinate rotation)
- [ ] Compressibility Effects (Prandtl Glauert corrector)

# Backends

- [X] ISPC
- [X] CUDA

# Maintainability

- [X] Clangd
- [X] Clang-Format
- [X] Clang-Tidy
- [X] Testing framework (via `xmake test`)
- [X] Compilation CI
- [ ] Documented code (doxygen style)

## Build and run

Requirements: 
- Windows (MSVC compiler) or Linux (GCC or Clang)
- [xmake](https://xmake.io/#/) build system
    - For Windows, the [installer](https://github.com/xmake-io/xmake/releases) is recommended
    - For Linux, run `curl -fsSL https://xmake.io/shget.text | bash`

Then build and run the default config:

```bash
xmake -y -v && xmake run
```

The solver should run and output the aerodynamic coefficients along with some time metrics:

```
reading plot3d mesh
number of panels: 4096
ns: 64
nc: 64
LHS: 12.106 ms
RHS: 4 us
Solve: 341.976 ms
Compute forces: 104 us
>>> Alpha: 5.0 | CL = 0.464887 CD = 0.005636 CMx = 1.157476 CMy = -0.017555 CMz = 0.101128
SOLVER RUN: 358.235 ms
```

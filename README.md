# TinyVLM

### Toy potential flow to experiment and learn high performance techniques
---

## Main features roadmap:

- [X] Stationary VLM
- [ ] Non-linear
- [ ] Panel-method
- [ ] Spectral domain

## Build and run

Requirements: 
- Windows (MSVC compiler) or Linux (GCC or Clang)
- CPU that supports AVX2
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

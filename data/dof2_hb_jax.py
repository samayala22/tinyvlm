import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jax
import jax.numpy as jnp
from functools import partial
from enum import Enum
from dataclasses import dataclass
import time
import numpy as np


# -----------------------------------------------------------------------------
jnp.set_printoptions(
    linewidth=200,  # max line width
    formatter={'float': '{:.3e}'.format}  # format shortE
)
jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
class NDVars:
    def from_dvars(self, d_vars):
        self.U = d_vars.u_inf / (self.omega_a * d_vars.b)
        self.mu = d_vars.m / (jnp.pi * d_vars.rho * d_vars.b**2)
        self.r_a = d_vars.r_a / d_vars.b
        self.x_a = d_vars.x_a / d_vars.b

    def __init__(self, a_h, omega, zeta_a, zeta_h, x_a, mu, r_a, U):
        self.a_h = a_h
        self.omega = omega
        self.zeta_a = zeta_a
        self.zeta_h = zeta_h
        self.x_a = x_a
        self.mu = mu
        self.r_a = r_a
        self.U = U

# -----------------------------------------------------------------------------
def alpha_freeplay(alpha, M0=0.0, Mf=0.0,
                   delta=jnp.radians(0.5),
                   a_f=jnp.radians(0.25)):
    if alpha < a_f:
        return M0 + alpha - a_f
    elif alpha <= a_f + delta:
        return M0 + Mf * (alpha - a_f)
    else:
        return M0 + alpha - a_f + delta * (Mf - 1)

def alpha_cubic(alpha, beta0=0.0, beta1=0.1,
                beta2=0.0, beta3=40.0):
    return beta0 + beta1*alpha + beta2*alpha**2 + beta3*alpha**3

def alpha_linear(alpha):
    return alpha

@dataclass
class System:
    M      : callable
    C      : callable
    K      : callable
    fnlt   : callable        # time‐domain NL force

def create_motion_system3():
    def fnlt(u, v, U_param):
        return jnp.array([
            - (ndv.omega / U_param)**2 * u[0] + v[1],
            - 1/(U_param**2) * torsional_func(u[1]) - v[0]
        ])
    
    def M(U_param):
        return jnp.array([
            [1.0, ndv.x_a],
            [ndv.x_a / ndv.r_a**2, 1.0]
        ])
    
    def C(U_param):
        return jnp.array([
            [2.0 * ndv.zeta_h * ndv.omega / U_param, 0.0],
            [0.0, 2.0 * ndv.zeta_a / U_param]
        ])
    
    def K(U_param):
        return jnp.zeros((2,2))

    return System(M, C, K, fnlt)

# -----------------------------------------------------------------------------
class Parametrisation(Enum):
    Local     = 1
    ArcLength = 2

# -----------------------------------------------------------------------------
def nabla(H):
    """Build the  (2H+1)x(2H+1) block matrix in a functional way."""
    size = 2*H + 1
    N = jnp.zeros((size, size), dtype=jnp.float64)
    block = jnp.array([[0.0,1.0],[-1.0,0.0]], dtype=jnp.float64)
    for j in range(1, H+1):
        i0 = 2*j - 1
        i1 = 2*j + 1
        N = N.at[i0:i1, i0:i1].set(j * block)
    return N

# -----------------------------------------------------------------------------
def hb_residual(X, *args):
    Omega = X[-1]
    param = X[-2]
    sys   = create_motion_system3()
    M = sys.M(param)
    C = sys.C(param)
    K = sys.K(param)

    nab = nabla(H)
    Z = (jnp.kron(Omega**2 * nab @ nab, M)
       + jnp.kron(Omega * nab,       C)
       + jnp.kron(jnp.eye(2*H+1),     K))
    R_lin = Z @ X[:-2]

    T       = 2.0*jnp.pi / Omega
    dt      = T / n_samples
    Xc_real = X[:-2].reshape((n_coeffs, n_dofs)).T
    Xc      = X_to_complex(Xc_real)

    # AFT → time domain
    k_arr = jnp.arange(H+1, dtype=Xc.dtype)
    q     = jnp.fft.irfft(Xc,      n_samples, axis=1, norm='forward')
    q_dot = jnp.fft.irfft(1j*Omega*k_arr[None,:]*Xc,
                          n_samples, axis=1, norm='forward')
    
    gamma, gamma_inv = build_gamma_op(Omega)
    jax.debug.print("{}", jnp.allclose(q.T.reshape(-1), gamma @ X[:-2]))

    t = jnp.arange(n_samples, dtype=X.dtype) * dt

    def single_step(ti, qi, qdi):
        return -sys.fnlt(qi, qdi, param)

    R_nlt = jax.vmap(single_step,
                     in_axes=(0,1,1),
                     out_axes=1)(t, q, q_dot)

    R_nl_fft = jnp.fft.rfft(R_nlt,
                             n_samples,
                             axis=1,
                             norm='backward')
    R_nl_fft = R_nl_fft[:, :H+1] / n_samples

    R_real = X_to_real(R_nl_fft)
    return R_lin + R_real.T.reshape((-1,))

def hb_residual2(X, *args):
    Omega = X[-1]
    param = X[-2]
    sys   = create_motion_system3()
    M = sys.M(param)
    C = sys.C(param)
    K = sys.K(param)

    T       = 2.0*jnp.pi / Omega
    dt      = T / n_samples
    Xc_real = X[:-2].reshape((n_coeffs, n_dofs)).T
    Xc      = X_to_complex(Xc_real)

    # AFT → time domain
    k_arr = jnp.arange(H+1, dtype=Xc.dtype)
    q     = jnp.fft.irfft(Xc,      n_samples, axis=1, norm='forward')
    q_dot = jnp.fft.irfft(1j*Omega*k_arr[None,:]*Xc,
                          n_samples, axis=1, norm='forward')

    t = jnp.arange(n_samples, dtype=X.dtype) * dt

    def single_step(ti, qi, qdi):
        return -sys.fnlt(qi, qdi, param)

    R_nlt = jax.vmap(single_step,
                     in_axes=(0,1,1),
                     out_axes=1)(t, q, q_dot)

    R_nl_fft = jnp.fft.rfft(R_nlt,
                             n_samples,
                             axis=1,
                             norm='backward')
    R_nl_fft = R_nl_fft[:, :H+1] / n_samples

    R_real = X_to_real(R_nl_fft)
    return R_real.T.reshape((-1,))

def build_dft(H: int, N: int, omega: float) -> jnp.ndarray:
    T = 2 * jnp.pi / omega
    t = jnp.arange(N, dtype=jnp.float64) * (T / N)

    rows = [jnp.full((N,), 0.5, dtype=jnp.float64)]

    for m in range(1, H + 1):
        rows.append(jnp.cos(m * omega * t))
        rows.append(jnp.sin(m * omega * t))

    dft = jnp.stack(rows, axis=0)
    return dft

# def build_gamma_op(omega):
#     dft = build_dft(H, n_samples, omega)
#     gamma = jnp.zeros((n_dofs * n_samples, n_dofs * n_coeffs))
#     gamma_m1 = jnp.zeros((n_dofs * n_coeffs, n_dofs * n_samples))
#     I_n = jnp.eye(n_dofs, dtype=jnp.float64)

#     for i in range(n_dofs):
#         gamma = gamma.at[:, i*n_dofs:(i+1)*n_dofs].set(jnp.kron(I_n, dft[i, :].reshape((n_samples, 1))))
#         gamma_m1 = gamma_m1.at[i*n_dofs:(i+1)*n_dofs, :].set(jnp.kron(I_n, dft[i, :]))

#     gamma_m1 = gamma_m1.at[:n_dofs, :].set(gamma_m1[:n_dofs, :] * 2.0)
#     gamma_m1 = (2 / n_samples) * gamma_m1
#     return gamma, gamma_m1

# def build_gamma_op(omega):
#     # time samples
#     N = n_samples
#     t = jnp.arange(N) * (2.0*jnp.pi/N)

#     # forward basis (rows = 0:DC, 1…2H = cos1, sin1, …, cosH, sinH)
#     TH = [ 0.5 * jnp.ones(N) ]
#     for k in range(1, H+1):
#         TH.append( jnp.cos( k*t ) )
#         TH.append( jnp.sin( k*t ) )
#     TH = jnp.stack(TH, axis=0)   # shape = (2H+1, N)

#     # build gamma  : maps Fourier→time
#     #   x_time = (I_n ⊗ THᵀ)  x_coeff
#     gamma = jnp.kron( jnp.eye(n_dofs), TH.T )

#     # build gamma⁻¹: maps time→Fourier
#     #   x_coeff = (I_n ⊗ TH) * (2/N)  x_time, except the DC row is 1/N
#     invTH = TH.at[0].set( 1.0 * jnp.ones(N) )  # temporarily DC=1
#     gamma_m1 = jnp.kron( jnp.eye(n_dofs), invTH )
#     gamma_m1 = gamma_m1 * (2.0 / N)
#     # now fix the DC row back to 1/N instead of 2/N
#     gamma_m1 = gamma_m1.at[:, 0:n_dofs].multiply(0.5)
#     return gamma, gamma_m1

def build_gamma_op(omega):
    """
    Returns
      Γ        :  (n_dofs*N) × (n_dofs*(2H+1))   real  —  maps Fourier→time
      Γ_inv    :  (n_dofs*(2H+1)) × (n_dofs*N)   real  —  maps time→Fourier
    so that
      q_time = Γ   @ x_coeff
      x_coeff = Γ⁻¹ @ q_time

    and so that these two are exactly the forward/inverse you get
    from    irfft( rfft(·,n=N), n=N )   up to machine eps.
    """
    # 1) time‐grid over [0,2π)
    N = n_samples
    t = jnp.arange(N, dtype=jnp.float64) * (2*jnp.pi/N)

    # 2) build the real‐Fourier basis TH  of size (2H+1)×N
    #    row 0 =  constant ½
    #    rows 1,2 = cos(1*t), sin(1*t)
    #    rows 3,4 = cos(2*t), sin(2*t)
    #       ...
    TH_rows = [jnp.ones(N, dtype=jnp.float64)]
    for k in range(1, H+1):
        TH_rows.append(jnp.cos(k * t))
        TH_rows.append(jnp.sin(k * t))
    TH = jnp.stack(TH_rows, axis=0)   # shape = (2H+1, N)

    # 3) build Γ = Iₙ ⊗ THᵀ
    #    so that if x_coeff is stacked as [dof0_coeffs; dof1_coeffs; ...],
    #    then q_time = (I ⊗ THᵀ) x_coeff  has shape (n_dofs*N).
    Gamma = jnp.kron(jnp.eye(n_dofs, dtype=jnp.float64), TH.T)

    # 4) build Γ⁻¹ = Iₙ ⊗ TH  scaled by (2/N), then fix the DC lines back to (1/N)
    #    so that the inverse reproduces exactly what a "rfft(...)/N" does.
    InvTH = TH.at[0].set(0.5 * jnp.ones_like(TH[0]))   # temporarily set DC row=1
    Gamma_inv = jnp.kron(jnp.eye(n_dofs, dtype=jnp.float64), InvTH) * (2.0 / N)

    return Gamma, Gamma_inv

# @jax.jit
def hb_jacobian(X, *args):
    """
    X          : stacked real HB‐coeffs + [param, Ω]
    TH         : precomputed basis of shape ((2H+1), n_samples)
    sys        : your System(...) with fnlt, etc.
    H,n_samples: HB‐settings
    """
    # unpack
    Om    = X[-1]
    param = X[-2]
    sys   = create_motion_system3()
    M = sys.M(param)
    C = sys.C(param)
    K = sys.K(param)
    dMdU = jax.jacobian(sys.M)(param)
    dCdU = jax.jacobian(sys.C)(param)
    dKdU = jax.jacobian(sys.K)(param)
    nab   = nabla(H)

    # ∂/∂x = Z
    L_lin = ( jnp.kron(Om**2 * (nab @ nab), M)
            + jnp.kron(   Om      * nab,       C)
            + jnp.kron(jnp.eye(2*H+1),         K) )

    nrows = X.shape[0] - 2
    ncols = X.shape[0]
    J     = jnp.zeros((nrows, ncols), dtype=X.dtype)

    J = J.at[:, :-2].set(L_lin)

    # ∂/∂param
    L_p = ( jnp.kron(Om**2 * (nab @ nab), dMdU)
          + jnp.kron(   Om      * nab,       dCdU)
          + jnp.kron(jnp.eye(2*H+1),         dKdU)
          ) @ X[:-2]
    J = J.at[:, -2].set(L_p)

    # ∂/∂Ω
    L_w = ( jnp.kron(2*Om * (nab @ nab), M)
          + jnp.kron(          nab       , C)
          ) @ X[:-2]
    J = J.at[:, -1].set(L_w)

    # Compute nonlinear force jacobian with respect to x
    jac_fnl_x = jax.jacobian(sys.fnlt, argnums=0)
    jac_fnl_xd = jax.jacobian(sys.fnlt, argnums=1)
    jac_fnl_param = jax.jacobian(sys.fnlt, argnums=2)

    T       = 2.0*jnp.pi / Om
    dt      = T / n_samples
    Xc_real = X[:-2].reshape((n_coeffs, n_dofs)).T
    Xc      = X_to_complex(Xc_real)

    # AFT → time domain
    k_arr = jnp.arange(H+1, dtype=Xc.dtype)
    q     = jnp.fft.irfft(Xc, n_samples, axis=1, norm='forward')
    q_dot = jnp.fft.irfft(1j*Om*k_arr[None,:]*Xc, n_samples, axis=1, norm='forward')
    param_vec = jnp.full((n_samples,), param, dtype=X.dtype)
    
    # Build the nxnxN tensors that hold the jacobians evaluated at the time samples
    Jac_fnl_x = jax.vmap(jac_fnl_x, in_axes=(1,1,0), out_axes=2)(q, q_dot, param_vec)
    Jac_fnl_xd = jax.vmap(jac_fnl_xd, in_axes=(1,1,0), out_axes=2)(q, q_dot, param_vec)
    Jac_fnl_param = jax.vmap(jac_fnl_param, in_axes=(1,1,0), out_axes=1)(q, q_dot, param_vec)

    Jac_fnl_x_mat = jnp.zeros((n_dofs * n_samples, n_dofs * n_samples))
    Jac_fnl_xd_mat = jnp.zeros((n_dofs * n_samples, n_dofs * n_samples))
    Jac_fnl_param_mat = jnp.zeros((n_dofs * n_samples, n_samples))
    gamma, gamma_m1 = build_gamma_op(Om)
    jax.debug.print("gamma {}", gamma)
    jax.debug.print("gamma_m1 {}", gamma_m1 @ gamma)

    for i in range(n_dofs):
        Jac_fnl_param_mat = Jac_fnl_param_mat.at[i*n_samples:(i+1)*n_samples, :].set(jnp.diag(Jac_fnl_param[i,:]))
        for j in range(n_dofs):
            Jac_fnl_x_mat = Jac_fnl_x_mat.at[i*n_samples:(i+1)*n_samples, j*n_samples:(j+1)*n_samples].set(jnp.diag(Jac_fnl_x[i,j,:]))
            Jac_fnl_xd_mat = Jac_fnl_xd_mat.at[i*n_samples:(i+1)*n_samples, j*n_samples:(j+1)*n_samples].set(jnp.diag(Jac_fnl_xd[i,j,:]))
    
    J_nl = jnp.zeros_like(J)
    J_nl = J_nl.at[:, :-2].set(gamma_m1 @ Jac_fnl_x_mat @ gamma + gamma_m1 @ Jac_fnl_xd_mat @ gamma @ jnp.kron(Om * nab, jnp.eye(n_dofs)))
    J_nl = J_nl.at[:, -2].set(gamma_m1 @ Jac_fnl_param.T.reshape(-1))
    J_nl = J_nl.at[:, -1].set(gamma_m1 @ Jac_fnl_xd_mat @ gamma @ jnp.kron(nab, jnp.eye(n_dofs)) @ X[:-2])
    
    # J_nl = jax.jacobian(hb_residual2)(X, *args)
    return J + J_nl
 

# -----------------------------------------------------------------------------
def extended_residual_jacobian_analytical(
    X, X_ref, z_ref,
    parametrisation,
    *args
):
    N = X.shape[0]
    Jext = jnp.zeros((N, N), dtype=X.dtype)

    J_lin = hb_jacobian(X, *args)
    Jext = Jext.at[:-2, :].set(J_lin)

    # orthogonality/phase row
    Xmat = X_ref[:-2].reshape(2*H+1, n_dofs).T
    for k in range(1, H+1):
        c1 = (2*k - 1) * n_dofs
        c2 = (2*k    ) * n_dofs
        Jext = Jext.at[-2, c1:c1+n_dofs].set(  k   * Xmat[:, 2*k    ] )
        Jext = Jext.at[-2, c2:c2+n_dofs].set( -k   * Xmat[:, 2*k - 1] )
    Jext = Jext.at[-2, -2:].set(0.0)

    # continuation row
    if parametrisation is Parametrisation.Local:
        Jext = Jext.at[-1, :].set(z_ref)
    else:
        Jext = Jext.at[-1, :].set(2.0 * (X - X_ref))

    return Jext

def extended_residual(
    X,
    X_ref,
    z_ref,
    residual_func,
    parametrisation: Parametrisation,
    *args
):
    """
    Fully AD-based residual + Jacobian via jax.jacobian(extended_residual).
    """
    ext_res = jnp.zeros_like(X)
    ext_res = ext_res.at[:-2].set(residual_func(X, *args))

    # Integral orthogonality / phase condition
    X_mat     = X[:-2].reshape(2*H+1, n_dofs).T
    X_mat_ref = X_ref[:-2].reshape(2*H+1, n_dofs).T
    orthogonality = 0
    for k in range(1, H+1):
        orthogonality = orthogonality + \
            k * (jnp.dot(X_mat_ref[:, 2*k],     X_mat[:, 2*k-1])
               - jnp.dot(X_mat_ref[:, 2*k-1],   X_mat[:, 2*k]))
    ext_res = ext_res.at[-2].set(orthogonality)

    match parametrisation:
        case Parametrisation.Local:
            ext_res = ext_res.at[-1].set(jnp.dot(z_ref, X - X_ref))
        case Parametrisation.ArcLength:
            ext_res = ext_res.at[-1].set(jnp.dot(X - X_ref, X - X_ref) - ds**2)

    return ext_res

# -----------------------------------------------------------------------------
def create_lanczos_filter(N: int, m: float = 1.0) -> jnp.ndarray:
    h = jnp.arange(N, dtype=jnp.float64)
    x = h / (N + 1.0)
    L = jnp.where(h == 0.0,
                  1.0,
                  jnp.sinc(x) ** m)
    return L

def X_to_complex(X: jnp.ndarray) -> jnp.ndarray:
    dofs, _ = X.shape
    A0 = X[:, :1].astype(jnp.complex128)
    A_h = X[:, 1::2]
    B_h = X[:, 2::2]
    harmonics = (A_h - 1j*B_h) / 2.0
    return jnp.concatenate([A0, harmonics], axis=1)

def X_to_real(X: jnp.ndarray, lanczos_m: float = 1.0) -> jnp.ndarray:
    dofs, N = X.shape
    L = create_lanczos_filter(N, m=lanczos_m)
    X0 = X[:, :1].real
    R  = X[:, 1:].real
    I  = X[:, 1:].imag
    window = L[1:]
    cR =  2.0 * window
    cI = -2.0 * window
    R_scaled = R * cR
    I_scaled = I * cI
    tail = jnp.stack([R_scaled, I_scaled], axis=-1).reshape(dofs, -1)
    return jnp.concatenate([X0, tail], axis=1)

def numerical_jac_lax(f, x, *args):
    n = x.shape[0]
    eps = jnp.finfo(x.dtype).eps
    def body_fun(j, jac):
        h = jnp.maximum(jnp.cbrt(eps) * jnp.abs(x[j]), eps)
        xp = x.at[j].add(h)
        xm = x.at[j].add(-h)
        delta = xp[j] - xm[j]
        col = (f(xp, *args) - f(xm, *args)) / delta
        return jac.at[:,j].set(col)
    jac0 = jnp.zeros((n,n), x.dtype)
    return jax.lax.fori_loop(0, n, body_fun, jac0)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Choose your torsional spring law:
    torsional_spring = 1
    if torsional_spring == 0:
        torsional_func = alpha_freeplay
    elif torsional_spring == 1:
        torsional_func = alpha_cubic
    else:
        torsional_func = alpha_linear

    # Harmonic‐balance settings
    H        = 3
    n_harmonics = H
    n_dofs   = 2
    n_coeffs = 2*H + 1
    n_samples= (H+1) * (2**4)

    # Parameters
    flutter_speed       = 6.285
    flutter_ratio_start = 0.3
    ds                  = 0.05

    param_start = flutter_speed * flutter_ratio_start
    param_end   = 4.5

    # nondimensional variables
    ndv = NDVars(
        a_h    = -0.5,
        omega  = 0.2,
        zeta_a = 0.1,
        zeta_h = 0.1,
        x_a    = 0.25,
        mu     = 100.0,
        r_a    = 0.5,
        U      = param_start
    )

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)
    X0    = jax.random.uniform(keys[0], shape=(n_dofs*(2*H+1) + 2,))
    X_ref = jax.random.uniform(keys[1], shape=X0.shape)
    z_ref = jax.random.uniform(keys[2], shape=X0.shape)

    # -----------------------------------------------------------------------------
    # 1) Half‐analytic (hb_jacobian) + small AD for nonlin → extended_residual_jacobian2
    # 2) Full AD → jax.jacobian(extended_residual)
    # 3) Finite difference
    #
    # We'll JIT‐compile (1) and (2) and then benchmark all three.

    # JIT‐compile the two JAX‐based routines
    J2_jit = jax.jit(
        extended_residual_jacobian_analytical,
        static_argnums=(3)     # freeze residual_func and parametrisation
    )
    J3_jit = jax.jit(
        jax.jacobian(extended_residual),
        static_argnums=(3,4)
    )

    numerical_jac_jit = jax.jit(numerical_jac_lax, static_argnums=(0, 4, 5, 6, 7))

    # Warm‐up / compile
    j2 = J2_jit(X0, X_ref, z_ref, Parametrisation.ArcLength, True, False)
    j3 = J3_jit(X0, X_ref, z_ref, hb_residual, Parametrisation.ArcLength, True, False)
    j4 = numerical_jac_jit(extended_residual, X0, X_ref, z_ref,
                      hb_residual, Parametrisation.ArcLength, True, False)

    print(j2)
    print(j3)
    np.testing.assert_allclose(j2, j3)
    np.testing.assert_allclose(j2, j4, atol=1e-7)
    np.testing.assert_allclose(j3, j4, atol=1e-7)

    # small helper
    def benchmark(func, *fargs, n_runs=5):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = func(*fargs)
        t1 = time.perf_counter()
        return (t1 - t0)/n_runs

    # run the timings
    runs = 100
    t2 = benchmark(
        J2_jit,
        X0, X_ref, z_ref,
        Parametrisation.ArcLength,
        True, False,
        n_runs=runs
    )
    t3 = benchmark(
        J3_jit,
        X0, X_ref, z_ref,
        hb_residual, Parametrisation.ArcLength,
        True, False,
        n_runs=runs
    )
    t4 = benchmark(
        numerical_jac_jit,
        extended_residual,
        X0, X_ref, z_ref,
        hb_residual, Parametrisation.ArcLength,
        True, False,
        n_runs=runs
    )

    print(f"\nBenchmark (avg over {runs} runs):")
    print(f"  J2_jit (analytic+AD) : {t2:.4e} s")
    print(f"  J3_jit (full AD)     : {t3:.4e} s")
    print(f"  numerical_jac (FD)   : {t4:.4e} s")

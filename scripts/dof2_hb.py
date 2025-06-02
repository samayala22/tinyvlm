import sys

sys.path.append(r"C:\Users\samay\Documents\GitHub\tinyvlm\build\windows\x64\release")

import numpy as np
import autograd
import autograd.numpy as anp
import scipy as sp
import plotting as plot

import os
import subprocess
from enum import Enum
from dataclasses import dataclass

# local imports
import dof2
import vanderpol as vdp
import helpers
import continuation as cont

from libhbvlm import *

np.set_printoptions(
    linewidth=200, # max line width
    formatter={'float': '{:.3e}'.format} # format shortE
) 

def getenv(key):
    var = os.getenv(key)
    if not var or int(var) == 0:
        return False
    return True

EPS = np.finfo(np.float64).eps
def cd2_h(x0): return np.maximum(np.where(np.abs(x0) > 1, np.cbrt(EPS * np.abs(x0)), np.cbrt(EPS) * np.abs(x0)), EPS)
def fd_h(x0): return np.maximum(np.sqrt(EPS) * np.abs(x0), EPS)

class Parametrisation(Enum):
    Local = 1
    ArcLength = 2

def create_fourier_basis(omega, harmonics, t):
    unknowns = 2 * harmonics + 1
    basis = np.zeros((unknowns))
    dbasis = np.zeros((unknowns))
    ddbasis = np.zeros((unknowns))
    basis[0] = 1
    dbasis[0] = 0
    ddbasis[0] = 0
    for i in range(harmonics):
        k = float(i+1)
        basis[2 * i + 1] = np.cos(omega * t * k)
        basis[2 * i + 2] = np.sin(omega * t * k)
        dbasis[2 * i + 1] = - omega * k * np.sin(omega * t * k)
        dbasis[2 * i + 2] = omega * k * np.cos(omega * t * k)
        ddbasis[2 * i + 1] = - (omega * k)**2 * np.cos(omega * t * k)
        ddbasis[2 * i + 2] = - (omega * k)**2 * np.sin(omega * t * k)

    return basis, dbasis, ddbasis

@dataclass
class System:
    M      : callable
    C      : callable
    K      : callable
    fnlt   : callable        # time‐domain NL force
    fnlf   : callable        # frequency‐domain NL force

def create_motion_system() -> System:
    def fnlt(u, v, omega, U):
        return np.array([
            - (ndv.omega / U)**2 * u[0],
            - 1/(U**2) * torsional_func(u[1])
        ])

    def fnlf(X, omega, U):
        forces_t = np.zeros_like(X)
        hbvlm_run(omega, X, forces_t)
        forces_t[0, :] = - forces_t[0, :] / (np.pi * ndv.mu)
        forces_t[1, :] = (2.0 * forces_t[1, :]) / (np.pi * ndv.mu * ndv.r_a**2)
        return forces_t
    
    def M(U):
        return anp.array([
            [1.0, ndv.x_a],
            [ndv.x_a / ndv.r_a**2, 1.0]
        ])
    
    def C(U):
        return anp.array([
            [2.0 * ndv.zeta_h * ndv.omega / U, 0.0],
            [0.0, 2.0 * ndv.zeta_a / U]
        ])
    
    def K(U):
        return anp.zeros((2,2))
    
    return System(M, C, K, fnlt, fnlf)

def tangent_predictor(J, zref, Xref):
    """Compute tangent vector using Seydel's pivot strategy."""
    # 1. Determine pivot indices
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_changes = np.abs(zref) / np.maximum(np.abs(Xref), 1e-4)
    kk = np.argsort(-rel_changes)  # Descending order
    
    # 2. Try different pivots until success
    ztmp = None
    for k in kk:
        # 3. Create constraint vector
        c = np.zeros_like(Xref)
        c[k] = 1.0
        
        # 4. Build extended system
        J_red = J[:-1, :]  # Exclude last row (parameter derivative)
        A = np.vstack([J_red, c])
        b = np.concatenate([np.zeros(J_red.shape[0]), [1.0]])
        
        # 5. Solve with least-squares for numerical stability
        ztmp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        if not np.any(np.isnan(ztmp)):
            break
    
    # 6. Normalize tangent vector
    # z = ztmp / np.linalg.norm(ztmp) # length 1 vector
    return ztmp

# def tangent_predictor(J, zref, Xref):
#     Q, R = np.linalg.qr(J.T)
#     z = Q[:, -1]
#     return z / np.linalg.norm(z)

def numerical_jac(f, x, method="3-point"):
    """
    Numerical Jacobian using precise central differences
    Performs 2*N evaluations of the function
    """
    m = len(f(x))
    n = len(x)

    jac = np.zeros((m, n))
    if method == "3-point":
        for j in range(n):
            h = cd2_h(x[j])
            # print(f"{j}| h: {h:.5e}")
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            delta = xp[j] - xm[j] # delta representable fp number
            jac[:, j] = (f(xp) - f(xm)) / delta
    elif method == "2-point":
        yj = f(x)
        for j in range(n):
            h = fd_h(x[j])
            xp = x.copy()
            xp[j] += h
            delta = xp[j] - x[j]
            jac[:, j] = (f(xp) - yj) / delta
    
    return jac

def hb_timedomain(t_begin, t_end, dt, dofs, X, omega, harmonics):
    # Plot the result in time domain
    samples = 2*harmonics+1
    vec_t = np.arange(t_begin, t_end + dt, dt)
    sol = np.zeros((3*dofs, vec_t.shape[0])) # u, v, a
    uf_sol_ = X.reshape(samples, dofs).T
    # uf_sol_ = xf0.reshape(samples, dofs).T
    for i, t in enumerate(vec_t):
        b, db, ddb = create_fourier_basis(omega, harmonics, t)
        sol[0:dofs, i] = uf_sol_ @ b
        sol[dofs:2*dofs, i] = uf_sol_ @ db
        sol[2*dofs:3*dofs, i] = uf_sol_ @ ddb

    return vec_t, sol

def extended_residual(
    X, # scaled
    X_ref,
    z_ref,
    residual_func,
    Dscale,
    parametrisation: Parametrisation,
    *args
):
    X_unscaled = X * Dscale # unscaled X
    ext_res = np.zeros_like(X)
    ext_res[:-2] = residual_func(X_unscaled, *args)

    # Integral orthogonality phase condition
    X_mat = X_unscaled[:-2].reshape(2*H+1, n_dofs).T
    X_ref_unscaled = X_ref * Dscale
    X_mat_ref = X_ref_unscaled[:-2].reshape(2*H+1, n_dofs).T
    orthogonality = 0
    for k in range(1, H+1):
        orthogonality += k * (np.dot(X_mat_ref[:, 2*k], X_mat[:, 2*k-1]) - np.dot(X_mat_ref[:, 2*k-1], X_mat[:, 2*k]))
    ext_res[-2] = orthogonality

    match parametrisation:
        case Parametrisation.Local:
            ext_res[-1] = np.dot(z_ref, X - X_ref)
        case Parametrisation.ArcLength:
            ext_res[-1] = np.dot(X - X_ref, X - X_ref) - ds**2 # iteration on a normal plane, perpendicular to tangent

    return ext_res

def nabla(H):
    nabla = np.zeros((2*H+1, 2*H+1))
    nabla_j = np.array([[0, 1], [-1, 0]])
    for j in range(1, H+1):
        nabla[2*j-1:2*j+1, 2*j-1:2*j+1] = j*nabla_j
    return nabla

def hb_jacobian(X, *args):
    Om = X[-2]
    param = X[-1]
    sys = create_motion_system()
    M = sys.M(param)
    C = sys.C(param)
    K = sys.K(param)
    dCdU = autograd.jacobian(sys.C)(param)
    dMdU = autograd.jacobian(sys.M)(param)
    dKdU = autograd.jacobian(sys.K)(param)

    nab = nabla(H)

    J_lin = np.zeros((X.shape[0]-2, X.shape[0]))

    J_lin[:, :-2] = np.kron(Om**2 * nab @ nab, M) + np.kron(Om * nab, C) + np.kron(np.eye(2*H+1), K)
    J_lin[:, -2] = (np.kron(Om**2 * nab @ nab, dMdU) + np.kron(Om * nab, dCdU) + np.kron(np.eye(2*H+1), dKdU)) @ X[:-2]
    J_lin[:, -1] = (np.kron(2*Om * nab @ nab, M) + np.kron(nab, C)) @ X[:-2]
    
    J_nlin = numerical_jac(hb_nonlinear_residual, X)
    return J_lin + J_nlin

def extended_residual_jacobian_hybrid(X, X_ref, z_ref, residual_func, Dscale, parametrisation, *args):
    Jext = np.zeros((X.shape[0], X.shape[0]))
    Jext[:-2, :] = hb_jacobian(X * Dscale, *args)
    
    # Integral orthogonal phase condition
    X_ref_unscaled = X_ref * Dscale
    X_mat_ref = X_ref_unscaled[:-2].reshape(2*H+1, n_dofs).T
    for k in range(1, H + 1):
        col_2k_minus_1 = int((2*k - 1) * n_dofs)
        col_2k = int(2*k * n_dofs)
        Jext[-2, col_2k_minus_1:col_2k_minus_1+n_dofs] = k * X_mat_ref[:, 2*k]
        Jext[-2, col_2k:col_2k + n_dofs] = - k * X_mat_ref[:, 2*k - 1]
    Jext[-2, -2:] = 0.0

    # Parametrisation
    match parametrisation:
        case Parametrisation.Local:
            Jext[-1, :] = z_ref
        case Parametrisation.ArcLength:
            Jext[-1, :] = 2 * (X - X_ref)

    # Scaling
    Jext[:-1, :] = Jext[:-1, :] @ np.diag(Dscale)
    
    return Jext

def extended_residual_jacobian(X, X_ref, z_ref, residual_func, Dscale, parametrisation, *args):
    Jext = np.zeros((X.shape[0], X.shape[0]))
    Jext[:-2, :] = numerical_jac(residual_func, X * Dscale)
    
    # Integral orthogonal phase condition
    X_ref_unscaled = X_ref * Dscale
    X_mat_ref = X_ref_unscaled[:-2].reshape(2*H+1, n_dofs).T
    for k in range(1, H + 1):
        col_2k_minus_1 = int((2*k - 1) * n_dofs)
        col_2k = int(2*k * n_dofs)
        Jext[-2, col_2k_minus_1:col_2k_minus_1+n_dofs] = k * X_mat_ref[:, 2*k]
        Jext[-2, col_2k:col_2k + n_dofs] = - k * X_mat_ref[:, 2*k - 1]
    Jext[-2, -2:] = 0.0

    # Parametrisation
    match parametrisation:
        case Parametrisation.Local:
            Jext[-1, :] = z_ref
        case Parametrisation.ArcLength:
            Jext[-1, :] = 2 * (X - X_ref)

    # Scaling
    Jext[:-1, :] = Jext[:-1, :] @ np.diag(Dscale)
    
    return Jext

def hb_linear_residual(X, *args):
    Om = X[-2]
    param = X[-1]
    sys = create_motion_system()
    M = sys.M(param)
    C = sys.C(param)
    K = sys.K(param)

    R_lin = np.zeros(X.shape[0]-2)
    nab = nabla(H)
    Z = np.kron(Om**2 * nab @ nab, M) + np.kron(Om * nab, C) + np.kron(np.eye(2*H+1), K)
    R_lin = Z @ X[:-2]
    return R_lin

def hb_nonlinear_residual(X, *args):
    Om = X[-2]
    param = X[-1]
    sys = create_motion_system()

    # T = 2 * np.pi / Om      # period
    # dt = T / n_samples              # time step
    Xc_real = X[:-2].reshape(n_coeffs, n_dofs).T
    Xc = vdp.X_to_complex(Xc_real) # Each row for each dof, each col corresponds to the jth fourier coeffs (a0, a1, b1, ... aH, bH)
    k = np.arange(H+1)
    q = np.fft.irfft(Xc, n_samples, axis=1, norm='forward') # no scaling
    q_dot = np.fft.irfft(1j * Om * k * Xc, n_samples, axis=1, norm='forward')
    # q_ddot = np.fft.ifft(- (w**2) * Q_fft).real
    R_nlt = np.zeros((n_dofs, n_samples))
    
    for s in range(n_samples):
        # t_n = s * dt
        R_nlt[:, s] = - sys.fnlt(q[:, s], q_dot[:, s], Om, param)
    
    R_nl_fft = np.fft.rfft(R_nlt, n_samples, axis=1, norm='backward') # no scaling
    R_nl = vdp.X_to_real((R_nl_fft[:, :H+1]) / n_samples).T.reshape(-1)
    
    R_nlft = - sys.fnlf(Xc_real, Om, param)
    R_nlf_fft = np.fft.rfft(R_nlft, n_coeffs, axis=1, norm='backward') # no scaling
    R_nlf = vdp.X_to_real(R_nlf_fft / n_coeffs, 0).T.reshape(-1)

    return R_nl + R_nlf

def hb_residual(X, *args):
    return hb_linear_residual(X, *args) + hb_nonlinear_residual(X, *args)

def objective_function(x, X_ref, z_ref, hb_residual, Dscale, parametrisation, bool1, bool2):
    residuals = extended_residual(
        x, X_ref, z_ref, hb_residual, Dscale, parametrisation, bool1, bool2
    )
    return np.sum(residuals**2)  # Scalar output

@helpers.Timing(prefix="Solver:")
def solve_nonlinear_system(X0, X_ref, z_ref, Dscale, parametrisation):
    sol = sp.optimize.root(
        extended_residual,
        X0,
        args=(X_ref, z_ref, hb_residual, Dscale, parametrisation, True, True),
        method='hybr',
        jac=extended_residual_jacobian,
        tol = 1e-6,
    )

    Q = sol.fjac.T
    R = np.zeros_like(sol.fjac)
    R[np.triu_indices_from(R)] = sol.r
    jac = Q @ R
    # latest_jac = extended_residual_jacobian(sol.x, X_ref, z_ref, hb_residual, Dscale, parametrisation, True, True)
    # print(jac)
    # print(latest_jac)
    # np.testing.assert_allclose(jac, latest_jac, rtol=1e-6)

    # sol = sp.optimize.least_squares(
    #     extended_residual,
    #     X0,
    #     jac=extended_residual_jacobian,
    #     method='trf',
    #     ftol=1e-6,
    #     gtol=1e-6,
    #     xtol=1e-6,
    #     x_scale='jac',
    #     args=(X_ref, z_ref, hb_residual, Dscale, parametrisation, True, True),
    #     diff_step=1e-6
    # )

    # sol = sp.optimize.minimize(
    #     objective_function,  # Scalar objective function
    #     X0,
    #     method='BFGS',   # Recommended for bound-constrained minimization
    #     # jac=extended_residual_jacobian,       # Finite difference Jacobian
    #     tol = 1e-6,  # Function tolerance
    #     # options={
    #     #     'ftol': 1e-6,   # Function tolerance
    #     #     'gtol': 1e-6,   # Gradient tolerance
    #     #     'eps': 1e-6,     # Step size for finite differences
    #     #     'maxls': 50      # Max line search steps (prevents stalling)
    #     # },
    #     args=(X_ref, z_ref, hb_residual, Dscale, parametrisation, True, True)
    # )

    if not sol.success:
        print(f"Nonlinear solver failed: {sol.message}")
        exit(1)
    print(f"param: {sol.x[-1]:.3f}, omega: {sol.x[-2]:.3f}, nfev: {sol.get('nfev', None)}, njev: {sol.get('njev', None)}")
    return sol.x, jac

def continuation(param_start, param_end, ds, X0, max_steps = 5000, scaling=True):
    """
    Continuation for autonomous systems using the harmonic balance method
    """
    # print("Initial guess X0:", X0)

    X_mat = np.zeros((X0.shape[0], max_steps))

    if param_end > param_start:
        param_direction = 1
        direction = 1
    else:
        param_direction = -1
        direction = -1

    X_ref = X0.copy()
    X_old = X0.copy()
    z_ref = np.zeros_like(X0)
    z_ref[-1] = 1

    Dscale = np.ones_like(X0)
    Dscale_prev = Dscale.copy()

    # Initial step
    Xp, J = solve_nonlinear_system(X0, X_ref, z_ref, Dscale, Parametrisation.Local)
    X0 = Xp.copy()
    X_mat[:, 0] = X0
    np.save("build/continuation.npy", X_mat[:, :1])

    iteration = 1
    while iteration < max_steps:
        if scaling:
            Dscale_prev = Dscale.copy()
            Dscale = np.maximum(np.abs(X0 * Dscale_prev), np.ones_like(X0))
            Dscale[-2] = 1.0 # omega is not scaled
            Dscale[-1] = 1.0 # param is not scaled
            X_ref = X_ref * (Dscale_prev / Dscale)
            X0 = X0 * (Dscale_prev / Dscale)
            X_old = X_old * (Dscale_prev / Dscale)

        # J = extended_residual_jacobian(X0, X_ref, z_ref * Dscale_prev, hb_residual, Dscale, Parametrisation.ArcLength, True, False)
        ztmp = tangent_predictor(J @ np.diag(1 / Dscale_prev), z_ref * Dscale_prev, X_ref) / Dscale
        z = ztmp / np.linalg.norm(ztmp)

        # Take a step in the tangent direction ensuring to stay along the solution path
        if (iteration > 1) and np.dot(X0-X_old, direction*ds*z) < 0:
            direction *= -1

        # Parametrizaton params
        X_ref = X0.copy()
        z_ref = z.copy()

        # Predictor step
        Xp = X0 + direction*ds*z
        # Corrector step
        Xtmp, J = solve_nonlinear_system(Xp, X_ref, z_ref, Dscale, Parametrisation.ArcLength)

        X_old = X0.copy()
        X0 = Xtmp.copy()

        # History
        X_mat[:, iteration] = X0 * Dscale

        fig = plot.create_dofs_figure(["Heave", "Pitch"])
        hb_sol_t, hb_sol = hb_timedomain(0.0, 1000.0, dt, n_dofs, X_mat[:-2, iteration], X_mat[-2, iteration], H)
        plot.add_data_and_psd(fig, hb_sol_t, hb_sol[0, :], "HB-VLM", 1, 1, 1)
        plot.add_data_and_psd(fig, hb_sol_t, hb_sol[1, :], "HB-VLM", 3, 1, 1)
        plot.add_data_and_psd(fig, hb_sol_t, hb_sol[2, :], "HB-VLM", 1, 2, 1)
        plot.add_data_and_psd(fig, hb_sol_t, hb_sol[3, :], "HB-VLM", 3, 2, 1)
        param_str = f"{X_mat[-1, iteration]:.2f}".replace('.', '_')
        plot.fig_save(fig, f"build/continuation/cont_{iteration}_{param_str}")

        iteration += 1
        np.save("build/continuation.npy", X_mat[:, :iteration]) # save every iteration

        if (X0[-1] - param_end) * param_direction >= 0:
            print("Continuation reached the end")
            break

    return X_mat[:, :iteration]

def truncated_series_approximation(u_tr, H):
    N_tr = u_tr.shape[1]  # Number of time samples
    dc = np.mean(u_tr, axis=1)
    u_tr = u_tr - dc[:, None] # offset mean value to prevent spectral leakage
    window = np.hanning(N_tr)
    u_tr_windowed = u_tr * window[None, :]  # Multiply each DoF by the window
    zp_factor = 4                   # zero padding factor
    N_fft = zp_factor * N_tr        # New FFT length after padding
    U_fft = np.fft.fft(u_tr_windowed, n=N_fft, axis=1)
    norm_factor = window.sum()
    freqs = np.fft.fftfreq(N_fft, dt)
    pos = freqs > 0
    f_pos = freqs[pos]
    U_pos = U_fft[:, pos]
    ref_dof = 0
    amplitude_ref = np.abs(U_pos[ref_dof, :])
    i0_ref = np.argmax(amplitude_ref)
    f0 = f_pos[i0_ref]
    omega0 = 2 * np.pi * f0        # Base angular frequency
    print("Base frequency: {:.3f} rad/s".format(omega0))

    coeffs = np.zeros((n_dofs, 2 * H + 1))
    coeffs[:, 0] = dc
    for h in range(1, H + 1):
        target = h * f0
        idx = np.argmin(np.abs(f_pos - target))  # Find the closest frequency bin
        Y = U_pos[:, idx]
        # Multiply by 2 because of the use of a one-sided FFT (except the DC term)
        coeffs[:, 2 * h - 1] = 2 * np.real(Y) / norm_factor   # cosine coefficient
        coeffs[:, 2 * h]     = -2 * np.imag(Y) / norm_factor   # sine coefficient

    return coeffs, omega0

if __name__ == "__main__":
    torsional_spring = 1
    torsional_spring_names = ["Freeplay", "Cubic", "Linear"]

    if (torsional_spring == 0):
        torsional_func = dof2.alpha_freeplay
    elif (torsional_spring == 1):
        torsional_func = dof2.alpha_cubic
    else:
        torsional_func = dof2.alpha_linear

    # Independent params
    H = 5
    vars_b = 0.5 # half chord
    n_dofs = 2
    n_coeffs = 2*H+1
    n_samples = (H+1)*(2**4) # sampling points (needs to be power of 2)
    flutter_speed = 6.285
    flutter_ratio_start = 0.3
    flutter_ratio_end = 0.8
    ds = 0.01

    hbvlm_init(H, 1.0/vars_b)

    # assert ((H+1) & H) == 0 # H+1 should be a power of 2

    # Dependent params
    # param_start = flutter_speed * flutter_ratio_start
    # param_end = flutter_speed * flutter_ratio_end
    param_start = 5.0
    param_end = 6.0
    # Time integration
    t_final = 2000.0
    dt = 0.2
    ndv = dof2.NDVars(
        a_h = -0.5,
        omega = 0.2,
        zeta_a = 0.0,
        zeta_h = 0.0,
        x_a = 0.25,
        mu = 100.0,
        r_a = 0.5,
        U = param_start
    )

    y0 = np.array([0, np.radians(3), 0, 0, 0, 0]) # h, a, hd, ad, x1, x2
    system = dof2.create_monolithic_system(y0, ndv, torsional_func)
    sol = sp.integrate.solve_ivp(system, (0, t_final), y0, t_eval=np.arange(0, t_final, dt), method='RK45')
    
    idx_start = int(0.75 * len(sol.t))
    t_tr = sol.t[idx_start:]
    u_tr = sol.y[0:2, idx_start:]   # shape = (n_dofs, N_tr)

    u_coeffs, omega0 = truncated_series_approximation(u_tr, H)
    
    X0 = np.zeros(n_dofs * (2 * H + 1) + 2)
    X0[:-2] = u_coeffs.T.reshape(-1)
    X0[-2] = omega0
    X0[-1] = param_start

    # z_ref = np.zeros_like(X0)
    # z_ref[-1] = 1.0
    # J_fd = extended_residual_jacobian(X0, X0, z_ref, hb_residual, Parametrisation.Local)
    # J_hy = extended_residual_jacobian_hybrid(X0, X0, z_ref, Parametrisation.Local)
    
    # print("FD cond: ", np.linalg.cond(J_fd))
    # print("Hybrid cond: ", np.linalg.cond(J_hy))

    # np.testing.assert_allclose(J_fd, J_hy)
    
    continuation(param_start, param_end, ds, X0, 1, False)
    # continuation(param_start, param_end, ds, X0, 5000, False)
    
    if getenv("PLOT"):
        X_mat = np.load("build/continuation.npy")
        if X_mat.shape[1] == 1:
            hb_sol_t, hb_sol0 = hb_timedomain(0.0, 1000.0, dt, n_dofs, X_mat[:-2, 0], X_mat[-2, 0], H)
            fig = plot.create_dofs_figure(["Heave", "Pitch"])
            dof2.plot_uvlm(fig)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[0, :], "HB-VLM", 1, 1, 1)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[1, :], "HB-VLM", 3, 1, 1)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[2, :], "HB-VLM", 1, 2, 1)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[3, :], "HB-VLM", 3, 2, 1)
            
            param_str = f"{X_mat[-1, 0]:.1f}".replace('.', '_')
            plot.fig_save(fig, f"build/hbvlm/hbvlm0_{param_str}")
        else:
            cont.plot_hb_continuation("2DOF HB-VLM Continuation", H, X_mat)

# Notes:
# - Dimitriadis introduction to nonlinear aeroelasticity p333: jacobian sign changes during continuation
# - Reproduce fig 7.14 ?
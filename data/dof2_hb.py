import numpy as np
import scipy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import autograd

import os
import subprocess
from enum import Enum
from dataclasses import dataclass

# local imports
import dof2
import vanderpol as vdp
import finite_diff as fd
import helpers

np.set_printoptions(
    linewidth=200, # max line width
    formatter={'float': '{:.3e}'.format} # format shortE
) 

def getenv(key):
    var = os.getenv(key)
    if not var or int(var) == 0:
        return False
    return True

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

def run_hbvlm(omega, H, read_gamma, write_gamma, reset_logfile=False):
        executable_path = "./build/windows/x64/release/hbvlm.exe"
        cwd_path = "./build/windows/x64/release/"
        logfile_path = "build/windows/x64/release/dof2_hbvlm.log"

        if reset_logfile:
            with open(logfile_path, 'w') as logfile:
                pass
            return

        with open(logfile_path, 'a') as logfile:
            result = subprocess.run(
                [executable_path, f"{omega:.9f}", f"{H}", f"{int(read_gamma)}", f"{int(write_gamma)}"],
                cwd=cwd_path,
                stdout=logfile,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True
            )
        
        if result.returncode != 0:
            print(f"Command failed (return code {result.returncode}), see {logfile_path}")
            exit(1)

@dataclass
class System:
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    dMdU: np.ndarray
    dCdU: np.ndarray
    dKdU: np.ndarray
    fnlt: callable # function nonlinear time domain input/time domain output
    dfntldU: callable # function nonlinear time domain derivative
    fnlf: callable # function nonlinear frequency domain input/time domain output
    dfnlfdU: callable # function nonlinear frequency domain derivative

def create_motion_system(omega: float, U_param: float, read_gamma: bool=False, write_gamma: bool=False):
    # NLvib params
    def fnlt(t, u, v):
        return np.array([
            - (ndv.omega / U_param)**2 * u[0],
            - 1/(U_param**2) * torsional_func(u[1])
        ])
    
    def dfntldU(t, u, v):
        return np.array([
            2.0 * (ndv.omega / U_param)**3 * u[0],
            2.0 /(U_param**3) * torsional_func(u[1])
        ])
    
    # @helpers.Timing(prefix="HBVLM: ")
    def fnlf(X):
        np.save("build/windows/x64/release/kin_coeffs.npy", X)
        run_hbvlm(omega, H, read_gamma, write_gamma)
        coeffs = np.load("build/windows/x64/release/hbvlm_t.npy").astype(np.float64)
        coeffs[0, :] = - coeffs[0, :] / (np.pi * ndv.mu)
        coeffs[1, :] = (2.0 * coeffs[1, :]) / (np.pi * ndv.mu * ndv.r_a**2)
        # coeffs = np.zeros_like(coeffs)
        return coeffs
    
    def dfnlfdU(X):
        return np.zeros(2)
    
    M = np.array([
        [1.0, ndv.x_a],
        [ndv.x_a / (ndv.r_a**2), 1.0]
    ])
    dMdU = np.zeros_like(M)
    C_ = np.array([[2.0 * ndv.zeta_h * ndv.omega, 0], [0, 2.0 * ndv.zeta_a]])
    C = C_ / U_param
    dCdU = - C_ / U_param**2
    # K = np.array([
    #     [(ndv.omega/U_param)**2, 0],
    #     [0, 1/(U_param**2)]
    # ])
    K = np.zeros((2,2))
    dKdU = np.zeros_like(K)

    return System(M, C, K, dMdU, dCdU, dKdU, fnlt, dfntldU, fnlf, dfnlfdU)

# TODO: improve this disgusting function
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
    z = ztmp / np.linalg.norm(ztmp) # length 1 vector
    return z

# def tangent_predictor(J, zref, Xref):
#     Q, R = np.linalg.qr(J.T)
#     z = Q[:, -1]
#     return z / np.linalg.norm(z)

def numerical_jac(f, x):
    """
    Numerical Jacobian using precise central differences
    Performs 2*N evaluations of the function
    """
    n = len(x)
    jac = np.zeros((n, n))
    for j in range(n):
        h = max(fd.cd2_h(x[j]), 1e-5) # limiter because hbvlm is in single precision
        # print(f"{j}| h: {h:.5e}")
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        delta = xp[j] - xm[j]
        jac[:, j] = (f(xp) - f(xm)) / delta
        if np.allclose(jac[:, j], np.zeros(n)):
            print("Jacobian zero problem")
    
    print("Condition number: ", np.linalg.cond(jac))
    return jac

def numerical_jac2(f, x, m, n):
    """
    Numerical Jacobian using precise central differences
    Performs 2*N evaluations of the function
    """
    assert n == len(x)
    assert m == len(f(x))

    jac = np.zeros((m, n))
    for j in range(n):
        h = max(fd.cd2_h(x[j]), 1e-5) # limiter because hbvlm is in single precision
        # print(f"{j}| h: {h:.5e}")
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        delta = xp[j] - xm[j] # delta representable fp number
        jac[:, j] = (f(xp) - f(xm)) / delta
    
    print("Condition number: ", np.linalg.cond(jac))
    return jac

def plot_hb_timedomain(fig, t_begin, t_end, dt, dofs, X, omega, harmonics, label="HB"):
    if not getenv("PLOT"): 
        return
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
    
    for dof in range(dofs):
        fig.add_trace(
            go.Scatter(
                x=vec_t,
                y=sol[dof, :],
                name=f"{label} dof #{dof} (omega={omega:.3f})"
            ),
            row=(dof+1),
            col=1
        )

class Parametrisation(Enum):
    Local = 1
    ArcLength = 2

@helpers.Timing(prefix="Residual: ")
def extended_residual(
    X,
    X_ref,
    z_ref,
    residual_func,
    parametrisation: Parametrisation,
    *args
):
    ext_res = np.zeros_like(X)
    ext_res[:-2] = residual_func(X, *args)

    # Integral orthogonality phase condition
    X_mat = X[:-2].reshape(2*H+1, n_dofs).T
    X_mat_ref = X_ref[:-2].reshape(2*H+1, n_dofs).T
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

def extended_residual_jacobian(X, X_ref, z_ref, residual_func, parametrisation, *args):
    return numerical_jac(lambda _X: extended_residual(_X, X_ref, z_ref, residual_func, parametrisation, True, False), X)

def nabla(H):
    nabla = np.zeros((2*H+1, 2*H+1))
    nabla_j = np.array([[0, 1], [-1, 0]])
    for j in range(1, H+1):
        nabla[2*j-1:2*j+1, 2*j-1:2*j+1] = j*nabla_j
    return nabla

def hb_jacobian(X, *args):
    Om = X[-1]
    param = X[-2]
    sys = create_motion_system(Om, param, *args)
    nab = nabla(H)

    J = np.zeros((X.shape[0]-2, X.shape[0]))

    J[:, :-2] = np.kron(Om**2 * nab @ nab, sys.M) + np.kron(Om * nab, sys.C) + np.kron(np.eye(2*H+1), sys.K)
    J[:, -2] = (np.kron(Om**2 * nab @ nab, sys.dMdU) + np.kron(Om * nab, sys.dCdU) + np.kron(np.eye(2*H+1), sys.dKdU)) @ X[:-2]
    J[:, -1] = (np.kron(2*Om * nab @ nab, sys.M) + np.kron(nab, sys.C)) @ X[:-2]
    
    nl_jacobian = jax.jacobian(hb_residual2)
    J2 = nl_jacobian(X, *args)
    assert J2.shape == J.shape, f"J2 shape: {J2.shape}, J shape: {J.shape}"
    # J2 = numerical_jac2(hb_residual2, X, X.shape[0]-2, X.shape[0])
    J += J2
    return J

def extended_residual_jacobian2(X, X_ref, z_ref, residual_func, parametrisation, *args):
    Jext = np.zeros((X.shape[0], X.shape[0]))
    Jext[:-2, :] = hb_jacobian(X, *args)
    
    X_mat_ref = X_ref[:-2].reshape(2*H+1, n_dofs).T
    for k in range(1, H + 1):
        # Calculate indices for the vectorized X elements
        col_2k_minus_1 = int((2*k - 1) * n_dofs)
        col_2k = int(2*k * n_dofs)
        
        # Term 1: k * X_ref[:, 2k] for odd columns
        Jext[-2, col_2k_minus_1:col_2k_minus_1+n_dofs] = k * X_mat_ref[:, 2*k]
        
        # Term 2: -k * X_ref[:, 2k-1] for even columns
        Jext[-2, col_2k:col_2k + n_dofs] = - k * X_mat_ref[:, 2*k - 1]
    
    Jext[-2, -2:] = 0.0 # OIC doesnt take into account param or omega
    match parametrisation:
        case Parametrisation.Local:
            Jext[-1, :] = z_ref
        case Parametrisation.ArcLength:
            Jext[-1, :] = 2 * (X - X_ref)
    
    return Jext

def hb_residual(X, *args):
    """
    X[:-2]: dof*(2*H+1) Fourier coefficients of the system [X0, Xc1, Xs1, ... XcH, XsH]
    where Xx is [xx_0, xx_1, ... xx_M] with M = n_dofs
    X[-2]: Continuation parameter
    X[-1]: Fourier series base frequency
    """
    Om = X[-1]
    param = X[-2]
    sys = create_motion_system(Om, param, *args)
    R_lin = np.zeros(X.shape[0]-2)

    # Compute the linear forces in Fourier domain (kroenecker free formula)
    # R_lin[0:n_dofs] = K @ X[0:n_dofs]
    # for k in range(1, H+1):
    #     i = (2*k-1) * n_dofs
    #     R_lin[i:i+n_dofs] = (K - (k*Om)**2 * M) @ X[i:i+n_dofs] + k * Om * C @ X[i+n_dofs:i+2*n_dofs]
    #     R_lin[i+n_dofs:i+2*n_dofs] = - k * Om * C @ X[i:i+n_dofs] + (K - (k*Om)**2 * M) @ X[i+n_dofs:i+2*n_dofs]

    nab = nabla(H)
    Z = np.kron(Om**2 * nab @ nab, sys.M) + np.kron(Om * nab, sys.C) + np.kron(np.eye(2*H+1), sys.K)
    R_lin = Z @ X[:-2]

    # Optimized FFT version of AFT
    T = 2 * np.pi / Om      # period
    dt = T / n_samples              # time step
    Xc_real = X[:-2].reshape(n_coeffs, n_dofs).T
    Xc = vdp.X_to_complex(Xc_real) # Each row for each dof, each col corresponds to the jth fourier coeffs (a0, a1, b1, ... aH, bH)
    k = np.arange(H+1)
    q = np.fft.irfft(Xc, n_samples, axis=1, norm='forward') # no scaling
    q_dot = np.fft.irfft(1j * Om * k * Xc, n_samples, axis=1, norm='forward')
    # q_ddot = np.fft.ifft(- (w**2) * Q_fft).real
    R_nlt = np.zeros((n_dofs, n_samples))
    for s in range(n_samples):
        t_n = s * dt
        R_nlt[:, s] = - sys.fnlt(t_n, q[:, s], q_dot[:, s])
    R_nlft = - sys.fnlf(Xc_real)
    
    R_nl_fft = np.fft.rfft(R_nlt, n_samples, axis=1, norm='backward') # no scaling
    R_nlf_fft = np.fft.rfft(R_nlft, n_coeffs, axis=1, norm='backward') # no scaling
    R_nl = vdp.X_to_real((R_nl_fft[:, :H+1]) / n_samples).T.reshape(-1)
    R_nlf = vdp.X_to_real(R_nlf_fft / n_coeffs, 0).T.reshape(-1)

    return R_lin + R_nl + R_nlf

@helpers.Timing(prefix="Solver:")
def solve_nonlinear_system(X0, X_ref, z_ref, parametrisation, xtol=1e-5):
    Xp, info, ier, mesg =  sp.optimize.fsolve(
        extended_residual,
        X0,
        args=(X_ref, z_ref, hb_residual, parametrisation, True, True),
        fprime=extended_residual_jacobian,
        full_output=True,
        xtol = xtol,
    )
    if ier != 1:
        print(f"Nonlinear solver failed: {mesg}")
        exit(1)

    print(f"param: {Xp[-2]:.3f}, omega: {Xp[-1]:.3f}, nfev: {info['nfev']}, njev: {info['njev']}")

    return Xp

    # sol = sp.optimize.root(
    #     extended_residual,
    #     X0,
    #     args=(X_ref, z_ref, hb_residual, parametrisation, True, True),
    #     method='hybr',
    #     jac=extended_residual_jacobian,
    #     tol = xtol
    # )
    # if not sol.success:
    #     print(f"Nonlinear solver failed: {sol.message}")
    #     exit(1)

    # return sol.x

def continuation(param_start, param_end, ds, X0):
    """
    Continuation for autonomous systems using the harmonic balance method
    """
    print("Initial guess X0:", X0)

    X_mat = np.zeros((X0.shape[0], max_continuation_steps))
    max_continuation_steps = 5000

    if param_end > param_start:
        param_direction = 1
        direction = 1
    else:
        param_direction = -1
        direction = -1

    X_ref = X0.copy()
    X_old = X0.copy()
    z_ref = np.zeros_like(X0)
    z_ref[-2] = 1

    # Initial step
    Xp = solve_nonlinear_system(X0, X_ref, z_ref, Parametrisation.Local)
    X0 = Xp.copy()
    X_mat[:, 0] = X0

    iteration = 1
    while iteration < max_continuation_steps:
        J = extended_residual_jacobian(X0, X_ref, z_ref, hb_residual, Parametrisation.ArcLength, True, False)
        z = tangent_predictor(J, z_ref, X_ref)

        # Take a step in the tangent direction ensuring to stay along the solution path
        if (iteration > 1) and np.dot(X0-X_old, direction*ds*z) < 0:
            direction *= -1

        # Parametrizaton params
        X_ref = X0.copy()
        z_ref = z.copy()

        # Predictor step
        Xp = X0 + direction*ds*z
        # Corrector step
        Xtmp = solve_nonlinear_system(Xp, X_ref, z_ref, Parametrisation.ArcLength)

        X_old = X0.copy()
        X0 = Xtmp.copy()

        # History
        X_mat[:, iteration] = X0
        iteration += 1
        if (X0[-2] - param_end) * param_direction >= 0:
            print("Continuation reached the end")
            break

    np.save("build/continuation.npy", X_mat[:, :iteration])
    return X_mat[:, :iteration]

def truncated_series_approximation(u_tr, H):
    N_tr = u_tr.shape[1]  # Number of time samples
        # ----- 1. Apply Hann Window -----
    # Create a Hann (Hanning) window to taper the data and reduce spectral leakage
    dc = np.mean(u_tr, axis=1)
    u_tr = u_tr - dc[:, None]

    window = np.hanning(N_tr)       # shape = (N_tr,)
    u_tr_windowed = u_tr * window[None, :]  # Multiply each DoF by the window

    # ----- 2. Zero Padding -----
    # Zero-pad the windowed signal to increase frequency resolution.
    zp_factor = 4                   # Adjust the zero-padding factor as desired
    N_fft = zp_factor * N_tr        # New FFT length after padding

    # Compute the FFT on the windowed and zero-padded data along the time axis.
    U_fft = np.fft.fft(u_tr_windowed, n=N_fft, axis=1)

    # Normalization factor.
    norm_factor = window.sum()

    # ----- 3. Compute the Frequency Vector -----
    # Use N_fft so that the frequency resolution is improved.
    freqs = np.fft.fftfreq(N_fft, dt)
    # Restrict to positive frequencies (excluding 0 if desired)
    pos = freqs > 0
    f_pos = freqs[pos]             # Array of positive frequencies
    U_pos = U_fft[:, pos]          # FFT coefficients corresponding to positive frequencies

    # ----- 4. Identify the Base Frequency -----
    # Use a reference degree of freedom (here, dof 0) to pick the base frequency.
    ref_dof = 0
    amplitude_ref = np.abs(U_pos[ref_dof, :])
    i0_ref = np.argmax(amplitude_ref)
    f0 = f_pos[i0_ref]
    omega0 = 2 * np.pi * f0        # Base angular frequency
    print("Base frequency: {:.3f} rad/s".format(omega0))

    # ----- 5. Extract Fourier Coefficients -----
    # Arrange the Fourier coefficients into an array.
    # The 0 index holds the DC term. Then (2*h-1) and (2*h) hold cosine and sine terms respectively.
    coeffs = np.zeros((n_dofs, 2 * H + 1))
    # coeffs[:, 0] = np.real(U_fft[:, 0]) / norm_factor  # DC term
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
    H = 3
    n_dofs = 2
    n_coeffs = 2*H+1
    n_samples = (H+1)*(2**4) # sampling points (needs to be power of 2)
    flutter_speed = 6.285
    flutter_ratio_start = 0.3
    flutter_ratio_end = 0.8
    ds = 0.05

    # assert ((H+1) & H) == 0 # H+1 should be a power of 2

    # Dependent params
    param_start = flutter_speed * flutter_ratio_start
    # param_end = flutter_speed * flutter_ratio_end
    # param_start = 2.0
    param_end = 4.5
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
    X0[-2] = param_start
    X0[-1] = omega0

    # Initialize the gamma values
    run_hbvlm(0, 0, False, False, reset_logfile=True) # only reset logfile

    sys = create_motion_system(omega0, param_start, False, True) 
    R_nlft = sys.fnlf(X0[:-2].reshape(n_coeffs, n_dofs).T)
    
    # if (getenv("PLOT")):
        # fig2 = make_subplots(
        #     rows=n_dofs, cols=1,
        #     subplot_titles=[f"Dof {i+1}" for i in range(n_dofs)],
        # )
        # fig2.update_layout(title="2 DOF Force Response")
        
        # R_nlf_fft = np.fft.rfft(R_nlft, n_coeffs, axis=1, norm='backward')
        # R_nlf = vdp.X_to_real(R_nlf_fft / n_coeffs, 0).T.reshape(-1)
        # plot_hb_timedomain(fig2, t_tr[0], t_tr[-1], 0.1, n_dofs, R_nlf, omega0, H, "HB")
        # fig2.update_yaxes(tickformat='.2e')
        # fig2.show()

        # fig = make_subplots(
        #     rows=n_dofs, cols=1,
        #     subplot_titles=[f"Dof {i+1}" for i in range(n_dofs)],
        #     vertical_spacing=0.1,
        #     horizontal_spacing=0.08
        # )
        # fig.update_layout(title="2 DOF Aeroelastic Response")
        # plot_hb_timedomain(fig, t_tr[0], t_tr[-1], 0.1, n_dofs, X0[:-2], X0[-1], H, "FFT")
        
        # z_ref = np.zeros_like(X0)
        # z_ref[-2] = 1
        # Xpp = solve_nonlinear_system(X0, X0.copy(), z_ref, True)
        # plot_hb_timedomain(fig, t_tr[0], t_tr[-1], 0.1, n_dofs, Xpp[:-2], Xpp[-1], H, "HB")

        # for dof in range(n_dofs):
        #     fig.add_trace(
        #         go.Scatter(
        #             x=t_tr,  # x values for new line
        #             y=u_tr[dof, :],  # y values for new line
        #             name=f"Time integration dof {dof}",  # legend label
        #             line=dict(color='red')  # optional: customize line color
        #         ),
        #         row=(dof+1),
        #         col=1
        #     )
        # fig.show()

    # X_mat = continuation(param_start, param_end, ds, X0)
    
    # if getenv("PLOT"):
    #     fig = go.Figure()
    #     fig.update_layout(title=f"2DOF Aeroelasticd Response ({torsional_spring_names[torsional_spring]} Pitch)")
    #     for h in range(1, H+1):
    #         fig.add_trace(
    #             go.Scattergl(
    #                 x = X_mat[-2, :],
    #                 y = np.sqrt(X_mat[2*h-1, :]**2 + X_mat[2*h, :]**2),
    #                 name = f"Harmonic {h}",
    #                 mode = "lines+markers"
    #             )
    #         )
    #     fig.update_xaxes(title_text=r"$\bar{U}$", showgrid=True)
    #     fig.update_yaxes(title_text=r"$||H_{j}||^{2}$", showgrid=True)
    #     fig.write_html("build/continuation.html", include_mathjax='cdn')
    #     fig.show()
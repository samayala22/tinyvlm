import numpy as np
import scipy as sp
import plotly.graph_objects as go
import os
import subprocess

# local imports
import dof2
import vanderpol as vdp
import finite_diff as fd

np.set_printoptions(formatter={'float': '{:.4e}'.format}) # format shortE

def getenv(key):
    var = os.getenv(key)
    if int(var) == 0:
        return False
    return True

def create_lanczos_filter(N, m=1):
    L = np.zeros(N)
    L[0] = 1
    def sinc(x): return np.sin(np.pi * x) / (np.pi*x)
    for i in range(1, N+1):
        xi = i / (N+1)
        L[i-1] = sinc(xi) ** m
    return L

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

def create_motion_system(omega, U_param):
    # NLvib params
    def func_nl(t, u, v):
        return np.array([
            - (ndv.omega / U_param)**2 * u[0],
            - 1/(U_param**2) * torsional_func(u[1])
        ])
    
    def func_nl_freq(X):
        np.save("build/windows/x64/release/kin_coeffs.npy", X)
        subprocess.run(
            ["./build/windows/x64/release/hbvlm.exe", f"{omega}", f"{H}"],
            cwd="./build/windows/x64/release/"
        )
        coeffs = np.load("build/windows/x64/release/hbvlm_t.npy")
        coeffs[0, :] = - coeffs[0, :] / (np.pi * ndv.mu)
        coeffs[1, :] = (2.0 * coeffs[1, :]) / (np.pi * ndv.mu * ndv.r_a**2)
        # coeffs = np.zeros_like(coeffs)
        return coeffs
    
    M = np.array([
        [1.0, ndv.x_a],
        [ndv.x_a / (ndv.r_a**2), 1.0]
    ])
    C = np.array([
        [2.0*ndv.zeta_h*(ndv.omega/U_param), 0],
        [0, 2.0*ndv.zeta_a/U_param]
    ])
    # K = np.array([
    #     [(ndv.omega/U_param)**2, 0],
    #     [0, 1/(U_param**2)]
    # ])
    K = np.zeros((2,2))

    return M, C, K, func_nl, func_nl_freq

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
        # h = max(fd.cd2_h(x[j]), np.finfo(np.float32).eps) # temporary limiter because hbvlm is in single precision
        h = 1e-6
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        delta = xp[j] - xm[j]
        jac[:, j] = (f(xp) - f(xm)) / delta
        if np.allclose(jac[:, j], np.zeros(n)):
            print("Jacobian zero problem")
    
    return jac

def plot_hb_timedomain(fig, t_begin, t_end, dt, dofs, X, omega, harmonics):
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
                name=f"HB dof {dof}"
            )
        )

def extended_residual(X, X_ref, z_ref, residual_func, init: bool):
    ext_res = np.zeros_like(X)
    ext_res[:-2] = residual_func(X)

    # Integral orthogonality phase condition
    X_mat = X[:-2].reshape(2*H+1, n_dofs).T
    X_mat_ref = X_ref[:-2].reshape(2*H+1, n_dofs).T
    orthogonality = 0
    for k in range(1, H+1):
        orthogonality += k * (np.dot(X_mat_ref[:, 2*k], X_mat[:, 2*k-1]) - np.dot(X_mat_ref[:, 2*k-1], X_mat[:, 2*k]))
    ext_res[-2] = orthogonality
    # ext_res[-2] = X[2] # phase fixing condition

    if init: # local parametrization
        ext_res[-1] = np.dot(z_ref, X - X_ref)
    else: # arc-length parametrization
        ext_res[-1] = np.dot(X - X_ref, X - X_ref) - ds**2 # iteration on a normal plane, perpendicular to tangent
    return ext_res

def extended_residual_jacobian(X, X_ref, z_ref, residual_func, init: bool):
    return numerical_jac(lambda _X: extended_residual(_X, X_ref, z_ref, residual_func, init), X)

def continuation(H, param_start, param_end, ds=.01, X0=None):
    """
    Continuation for autonomous systems using the harmonic balance method
    """
    samples = int(2*H+1)
    dofs = 2

    N = (H+1)*(2**3) # sampling points (needs to be power of 2)
    # N = 4*H+1
    print("Sampling points:", N)
    def hb_residual(X):
        """
        X[:-2]: dof*(2*H+1) Fourier coefficients of the system [X0, Xc1, Xs1, ... XcH, XsH]
        where Xx is [xx_0, xx_1, ... xx_M] with M = dofs
        X[-2]: Continuation parameter
        X[-1]: Fourier series base frequency
        """
        Om = X[-1]
        param = X[-2]
        M, C, K, func_nl, func_nl_freq = create_motion_system(Om, param)
        R_lin = np.zeros(X.shape[0]-2)

        # Compute the linear forces in Fourier domain
        R_lin[0:dofs] = K @ X[0:dofs]
        for k in range(1, H+1):
            i = (2*k-1) * dofs
            R_lin[i:i+dofs] = (K - (k*Om)**2 * M) @ X[i:i+dofs] + k * Om * C @ X[i+dofs:i+2*dofs]
            R_lin[i+dofs:i+2*dofs] = - k * Om * C @ X[i:i+dofs] + (K - (k*Om)**2 * M) @ X[i+dofs:i+2*dofs]

        # Optimized FFT version of AFT
        T = 2 * np.pi / Om      # period
        dt = T / N              # time step
        Xc_real = X[:-2].reshape(samples, dofs).T
        Xc = vdp.X_to_complex(Xc_real) # Each row for each dof, each col corresponds to the jth fourier coeffs (a0, a1, b1, ... aH, bH)
        q = np.fft.irfft(Xc, N, axis=1, norm='forward') # no scaling
        # for tidx in range(N):
        #     t = tidx * dt
        #     q2 = Xc_real[0, 0]
        #     for h in range(0, H):
        #         k = h+1
        #         q2 += np.cos(Om * t * k) * Xc_real[0, 2*h+1]
        #         q2 += np.sin(Om * t * k) * Xc_real[0, 2*h+2]
        #     np.testing.assert_allclose(q2, q[0, tidx])

        k = np.arange(H+1)
        q_dot = np.fft.irfft(1j * Om * k * Xc, N, axis=1, norm='forward')
        # q_ddot = np.fft.ifft(- (w**2) * Q_fft).real
        R_nlt = np.zeros((dofs, N))
        for s in range(N):
            t_n = s * dt
            R_nlt[:, s] = - func_nl(t_n, q[:, s], q_dot[:, s])
        R_nlft = - func_nl_freq(Xc_real)
        
        R_nl_fft = np.fft.rfft(R_nlt, N, axis=1, norm='backward') # no scaling
        R_nlf_fft = np.fft.rfft(R_nlft, samples, axis=1, norm='backward') # no scaling
        R_nl = vdp.X_to_real((R_nl_fft[:, :H+1]) / N).T.reshape(-1)
        R_nlf = vdp.X_to_real(R_nlf_fft / samples).T.reshape(-1)
        # R_nlft2 = np.fft.irfft(vdp.X_to_complex(vdp.X_to_real(R_nlf_fft / samples)), samples, axis=1, norm='forward')
        # R_nlft3 = np.fft.irfft(vdp.X_to_complex(vdp.X_to_real(np.fft.rfft(R_nlft, N, axis=1, norm='backward') / N)), N, axis=1, norm='forward') # no scaling
        # print(R_nlft.shape, R_nlft2.shape)
        # np.testing.assert_allclose(R_nlft3, R_nlft)
        # np.testing.assert_allclose(R_nlft2, R_nlft)
        return R_lin + R_nl + R_nlf
    
    print("Initial guess X0:", X0)

    # Continuation framework
    max_continuation_steps = 5000
    ds_min = 1e-5

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

    res0 = hb_residual(X0)
    print("Initial spectral residual norm:", np.linalg.norm(res0))
    # print(np.concatenate(([res0[0]], res0[1::2]/2, res0[2::2]/2)))
    Xp, info, ier, mesg = sp.optimize.fsolve(
        extended_residual,
        X0,
        args=(X_ref, z_ref, hb_residual,True),
        fprime=extended_residual_jacobian,
        # epsfcn=1e-5,
        full_output=True,
        xtol = 1e-6,
    ) # initial step
    if ier != 1:
        print(f"Initial step failed: {mesg}")
        return
    
    print(Xp)
    # X0[:-1] = Xp.x[:-1] # TODO: check if we cant just copy the whole thing
    X0 = Xp.copy()

    X_mat = np.zeros((X0.shape[0], max_continuation_steps))
    X_mat[:, 0] = X0

    if getenv("PLOT"):
        fig2 = go.Figure()
        fig2.update_layout(title=f"2 dof (omega={X_mat[-1, 0]})")
        plot_hb_timedomain(fig2, 0.0, 4 * np.pi / X_mat[-1, 0], 0.02, dofs, X_mat[:-2, 0], X_mat[-1, 0], H)
        fig2.show()

    return
    
    iteration = 1
    while iteration < max_continuation_steps:
        J = numerical_jac(lambda X: extended_residual(X, X_ref, z_ref, hb_residual, False), X0)

        z = tangent_predictor(J, z_ref, X_ref)
        # print(np.concatenate(([z[0]], z[1:-1:2]/2, z[2:-1:2]/2, [z[-1]])))
        # return

        # Take a step in the tangent direction ensuring to stay along the solution path
        if (iteration > 1) and np.dot(X0-X_old, direction*ds*z) < 0:
            direction *= -1

        # Parametrizaton params
        X_ref = X0.copy()
        z_ref = z.copy()
        while 1:
            Xp = X0 + direction*ds*z
            Xtmp, info, ier, mesg = sp.optimize.fsolve(
                extended_residual,
                Xp,
                args=(X_ref, z_ref, hb_residual,False),
                fprime=extended_residual_jacobian,
                full_output=True,
                xtol = 1e-6
            )
            if ier == 1:
                break
            else:
                print(f"Solver failed: {mesg}")
                ds = ds * 0.5
                if (ds < ds_min):
                    print("Continuation failed, exiting") 
                    return

        X_old = X0.copy()
        X0 = Xtmp.copy()

        print(f"param: {X0[-2]:.3f}, omega: {X0[-1]:.3f}, ds: {ds:.2e}, nfev: {info['nfev']}")

        # history
        X_mat[:, iteration] = X0
        iteration += 1
        if (X0[-2] - param_end) * param_direction >= 0:
            print("Continuation reached the end")
            break

    if getenv("PLOT"):
        fig = go.Figure()
        fig.update_layout(title="Continuation")
        for h in range(1, H+1):
            fig.add_trace(
                go.Scattergl(
                    x = X_mat[-2, :iteration],
                    y = np.sqrt(X_mat[2*h-1, :iteration]**2 + X_mat[2*h, :iteration]**2),
                    name = f"Harmonic {h}",
                    mode = "lines+markers"
                )
            )
        fig.show()

        fig2 = go.Figure()
        fig2.update_layout(title=f"2 dof (omega={X_mat[-1, iteration-1]})")
        plot_hb_timedomain(fig2, 0.0, 4 * np.pi / X_mat[-1, iteration-1], 0.02, dofs, X_mat[:-2, iteration-1], X_mat[-1, iteration-1], H)
        fig2.show()

if __name__ == "__main__":
    torsional_spring = 1
    torsional_spring_names = ["Freeplay", "Cubic", "Linear"]

    if (torsional_spring == 0):
        torsional_func = dof2.alpha_freeplay
    elif (torsional_spring == 1):
        torsional_func = dof2.alpha_cubic
    else:
        torsional_func = dof2.alpha_linear

    # HB Continuation
    H = 7
    assert ((H+1) & H) == 0 # H+1 should be a power of 2
    flutter_speed = 6.285
    flutter_ratio_start = 0.3
    flutter_ratio_end = 0.5
    param_start = flutter_speed * flutter_ratio_start
    param_end = flutter_speed * flutter_ratio_end
    ds = 0.02
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
    N_tr = len(t_tr)
    n_dofs = 2

    # ----- 1. Apply Hann Window -----
    # Create a Hann (Hanning) window to taper the data and reduce spectral leakage
    window = np.hanning(N_tr)       # shape = (N_tr,)
    u_tr_windowed = u_tr * window[None, :]  # Multiply each DoF by the window

    # ----- 2. Zero Padding -----
    # Zero-pad the windowed signal to increase frequency resolution.
    zp_factor = 4                   # Adjust the zero-padding factor as desired
    N_fft = zp_factor * N_tr        # New FFT length after padding

    # Compute the FFT on the windowed and zero-padded data along the time axis.
    U_fft = np.fft.fft(u_tr_windowed, n=N_fft, axis=1)

    # Normalization factor.
    # You may include a dt scaling if necessary, e.g., norm_factor = dt * window.sum()
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
    coeffs[:, 0] = np.real(U_fft[:, 0]) / norm_factor  # DC term

    for h in range(1, H + 1):
        target = h * f0
        idx = np.argmin(np.abs(f_pos - target))  # Find the closest frequency bin
        Y = U_pos[:, idx]
        # Multiply by 2 because of the use of a one-sided FFT (except the DC term)
        coeffs[:, 2 * h - 1] = 2 * np.real(Y) / norm_factor   # cosine coefficient
        coeffs[:, 2 * h]     = -2 * np.imag(Y) / norm_factor   # sine coefficient

    # ----- 6. Assemble the Initial Guess Vector -----
    # For example, combining Fourier coefficients with additional parameters.
    X0 = np.zeros(n_dofs * (2 * H + 1) + 2)
    X0[:-2] = coeffs.T.reshape(-1)
    X0[-2] = param_start
    X0[-1] = omega0

    if (getenv("PLOT")):
        fig = go.Figure()
        fig.update_layout(title=f"2 dof (omega={omega0})")
        plot_hb_timedomain(fig, t_tr[0], t_tr[-1], 0.1, n_dofs, X0[:-2], X0[-1], H)

        for dof in range(n_dofs):
            fig.add_trace(
                go.Scatter(
                    x=t_tr,  # x values for new line
                    y=u_tr[dof, :],  # y values for new line
                    name=f"Time integration dof {dof}",  # legend label
                    line=dict(color='red')  # optional: customize line color
                )
            )
        fig.show()
    continuation(H, param_start, param_end, ds, X0)
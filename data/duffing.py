import numpy as np
import scipy as sp
import plotly.express as px
import plotly.graph_objects as go

import os

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
    
def anderson_acceleration(f, x0, k_max=100, tol_res=1e-6, m=3):
    assert m <= x0.shape[0]
    n = x0.shape[0]

    Xbuf = np.zeros((n, m))  # differences in iterates
    Gbuf = np.zeros((n, m))  # differences in residuals (g = f(x) - x)
    x_curr = np.zeros(n)
    x_new = np.zeros(n)
    g_curr = np.zeros(n)
    g_new = np.zeros(n)
    gamma = np.zeros(n)
    residual_history = []

    x_curr = x0.copy() # in reality x_curr actually initialzed to x0
    x_new = f(x_curr)              # x₁ = f(x₀)
    g_curr = x_new - x_curr     # g₀ = f(x₀) - x₀
    Xbuf[:, 0] = g_curr # x1 - x0
    x_curr = x_new.copy()
    x_new = f(x_curr)              # x₂ = f(x₁)
    g_new = x_new - x_curr        # g₁ = f(x₁) - x₁
    Gbuf[:, 0] = g_new - g_curr # g1 - g0
    g_curr = g_new.copy()
    
    k = 1
    while k < k_max and np.linalg.norm(g_curr) > tol_res:
        m_k = min(m, k)
        _, gamma, _ = sp.linalg.lapack.dgels(Gbuf[:, :m_k], g_curr)

        x_new = x_curr + g_curr - (Xbuf[:, :m_k] + Gbuf[:, :m_k]) @ gamma[:m_k]

        Xbuf[:, k % m] = x_new - x_curr
        x_curr = x_new.copy()
        x_new = f(x_curr)
        g_new = x_new - x_curr
        Gbuf[:, k % m] = g_new - g_curr
        g_curr = g_new.copy()
        k += 1
        residual_history.append(np.linalg.norm(g_curr))

    return x_curr, len(residual_history)

def newmark_beta_step(M, C, K, v_i, a_i, delta_F, dt, beta=1/4, gamma=1/2):
    x2 = 1
    x1 = gamma / (beta * dt)
    x0 = 1 / (beta * dt**2)
    xd0 = 1 / (beta * dt)
    xd1 = gamma / beta
    xdd0 = 1/(2*beta)
    xdd1 = - dt * (1 - gamma / (2*beta))

    K_eff = x0 * M + x1 * C + x2 * K
    F_eff = delta_F + M @ (xd0 * v_i + xdd0 * a_i) + C @ (xd1 * v_i + xdd1 * a_i)
    du = np.linalg.solve(K_eff, F_eff)
    dv = x1 * du - xd1 * v_i - xdd1 * a_i
    da = x0 * du - xd0 * v_i - xdd0 * a_i

    return du, dv, da

def nonlinear_newmark_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt):
    t_steps = int(t_final / dt) + 1
    n = u0.shape[0] # number of equations
    vec_t = np.arange(0, t_final + dt, dt)
    u = np.zeros((n, t_steps))
    v = np.zeros((n, t_steps))
    a = np.zeros((n, t_steps))
    f_curr = nonlinear_func(0.0, u0, v0)
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = np.linalg.solve(M, f_curr - C @ v0 - K @ u0)

    avg_iters = 0

    for i in range(0, t_steps-1):
        t = i*dt
        f_next = f_curr.copy()
        du = np.zeros(2)
        du_k = np.zeros(2) + 1
        iteration = 0
        while (np.linalg.norm(du_k - du) / len(du) > 1e-10) and (iteration < 100):
            du_k = du[:]
            du, dv, da = newmark_beta_step(M, C, K, v[:,i], a[:,i], f_next - f_curr, dt)

            u[:,i+1] = u[:,i] + du
            v[:,i+1] = v[:,i] + dv
            a[:,i+1] = a[:,i] + da

            f_next = nonlinear_func(t, u[:,i+1], v[:,i+1])
            iteration += 1
        avg_iters += iteration
        f_curr = f_next.copy()

    print(f"Average iterations: {avg_iters / t_steps:.2f}")

    return vec_t, u, v, a

def nonlinear_newmark_anderson_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt):
    t_steps = int((t_final + dt)/ dt)
    n = u0.shape[0] # number of equations
    vec_t = np.arange(0, t_final + dt, dt)
    f_curr = nonlinear_func(0.0, u0, v0)
    x = np.zeros((3*n, t_steps+1))
    x[0:n, 0] = u0
    x[n:2*n, 0] = v0
    x[2*n:3*n, 0] = np.linalg.solve(M, f_curr - C @ v0 - K @ u0)

    avg_iters = 0
    for i in range(0, t_steps):
        t = i*dt
        f_next = f_curr.copy()
        def newmark_fixed_point(x_k):
            nonlocal f_next, x, i
            x_k_next = x_k.copy()
            du, dv, da = newmark_beta_step(M, C, K, x[n:2*n,i], x[2*n:3*n,i], f_next - f_curr, dt)
            x_k_next[0:n] = x[0:n, i] + du
            x_k_next[n:2*n] = x[n:2*n, i] + dv
            x_k_next[2*n:3*n] = x[2*n:3*n, i] + da
            f_next = nonlinear_func(t, x_k_next[0:n], x_k_next[n:2*n])
            return x_k_next
        
        x[:, i+1], iteration = anderson_acceleration(newmark_fixed_point, x[:, i], 100, 1e-10, 3)
        avg_iters += iteration
        f_curr = f_next.copy()

    print(f"Average iterations: {avg_iters / t_steps:.2f}")

    return vec_t, x[0:n, :], x[n:2*n, :], x[2*n:3*n, :]
    

# NLvib params
# P = 0.18
# mu = 1
# zeta = 0.05
# kappa = 1
# gamma = 0.1

# Wikipedia params
# P = 0.29
# mu = 1
# zeta = 0.3
# kappa = -1.0
# gamma = 1.0

# Krack
# P = 1.0
# mu = 1.0
# zeta = 0.0063
# kappa = 1.0
# gamma = 0.1

# Guillot
P = 3.0
mu = 1.0
zeta = 0.05
kappa = 1.0
gamma = 1.0

def create_motion_system(omega=0.5):
    # NLvib params
    def nonlinear_func(t, u, v):
        return np.array([P * np.cos(omega * t) - gamma * u[0]**3])
    
    def hb_nl_forces(t, u, v, omega_):
        return np.array([P * np.cos(omega_ * t) - gamma * u[0]**3])
    
    M = np.array([[mu]])
    C = np.array([[zeta]])
    K = np.array([[kappa]])

    u0 = np.array([1.0])
    v0 = np.array([0.0])

    return M, C, K, u0, v0, nonlinear_func, hb_nl_forces

def integrate_duffing(t_final, dt, om_s):
    M, C, K, u0, v0, nonlinear_func, _ = create_motion_system(om_s)
    t, u, v, a = nonlinear_newmark_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt)
    # t, u, v, a = nonlinear_newmark_anderson_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt)

    return t, u

# TODO: improve this disgusting function
# def tangent_predictor(J, zref, Xref):
#     """Compute tangent vector using Seydel's pivot strategy."""
#     # 1. Determine pivot indices
#     with np.errstate(divide='ignore', invalid='ignore'):
#         rel_changes = np.abs(zref) / np.maximum(np.abs(Xref), 1e-4)
#     kk = np.argsort(-rel_changes)  # Descending order
    
#     # 2. Try different pivots until success
#     ztmp = None
#     for k in kk:
#         # 3. Create constraint vector
#         c = np.zeros_like(Xref)
#         c[k] = 1.0
        
#         # 4. Build extended system
#         J_red = J[:-1, :]  # Exclude last row (parameter derivative)
#         A = np.vstack([J_red, c])
#         b = np.concatenate([np.zeros(J_red.shape[0]), [1.0]])
        
#         # 5. Solve with least-squares for numerical stability
#         ztmp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
#         if not np.any(np.isnan(ztmp)):
#             print("Nan found in tangent predictor, changing pivot...")
#             break
    
#     # 6. Normalize tangent vector
#     z = ztmp / np.linalg.norm(ztmp) # length 1 vector
#     return z

def tangent_predictor(J, zref, Xref):
    Q, R = np.linalg.qr(J.T)
    z = Q[:, -1]
    return z / np.linalg.norm(z)

EPS = np.finfo(np.float64).eps
def cd2_h(x0):
    """
    Optimal step size for a first order central difference
    Improved from the original formulation in Numerical Recipes in C section 5.7
    """
    return np.maximum(np.where(np.abs(x0) > 1, np.cbrt(EPS * np.abs(x0)), np.cbrt(EPS) * np.abs(x0)), EPS)
def cd2(f, x0, h=None):
    """
    Central difference with roundoff-aware formulation (Numerical Recipes in C section 5.7)
    """
    if h is None:
        h = cd2_h(x0)
    return (f(x0 + h) - f(x0 - h)) / ((x0 + h) - (x0 - h))

def numerical_jac(f, x):
    """
    Numerical Jacobian using precise central differences
    Performs 2*N evaluations of the function
    """
    n = len(x)
    jac = np.zeros((n, n))
    for j in range(n):
        h = cd2_h(x[j])
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        delta = xp[j] - xm[j]
        jac[:, j] = (f(xp) - f(xm)) / delta
    return jac

def plot_hb_timedomain(t_begin, t_end, dt, dofs, X, omega, harmonics):
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

    fig = px.line(x=vec_t, y=sol[0, :], title=f"Duffing Oscillator (omega={omega})")
    return fig

def extended_residual(X, X_ref, z_ref, residual_func, init: bool):
    ext_res = np.zeros_like(X)
    ext_res[:-1] = residual_func(X)
    if init: # local parametrization
        ext_res[-1] = np.dot(z_ref, X - X_ref)
    else: # arc-length parametrization
        ext_res[-1] = np.dot(X - X_ref, X - X_ref) - ds**2 # iteration on a normal plane, perpendicular to tangent
    return ext_res

def X_to_complex(X):
    """
    Converts a dofs * (2H+1) real array [A0, A1, B1, ... A_H, B_H] to a dofs * (H+1) comlex array
    """
    assert len(X.shape) == 2 # matrix form
    dofs = X.shape[0]
    H = int((X.shape[1] - 1) / 2)
    Xc = np.zeros((dofs, H+1), dtype=np.complex128)
    for d in range(dofs):
        Xc[d, 0] = X[d, 0] - 0j
    for h in range(1, H+1):
        for d in range(dofs):
            Xc[d, h] = (X[d, 2*h-1] - 1j * X[d, 2*h])/2
    return Xc

def X_to_real(X):
    """
    Converts a dofs * N complex array to a dofs * (2*N-1) real array
    """
    assert len(X.shape) == 2 # matrix form
    dofs = X.shape[0]
    N = X.shape[1]
    L = create_lanczos_filter(N, 0)
    Xr = np.zeros((dofs, 2*N-1), dtype=np.float64)
    for d in range(dofs):
        Xr[d, 0] = X[d, 0].real
    for h in range(1, N):
        for d in range(dofs):
            Xr[d, 2*h-1] = L[h] * 2 * X[d, h].real
            Xr[d, 2*h] = L[h] * -2 * X[d, h].imag
    
    return Xr

def hb_duffing(H, omega_start, omega_end, ds=.01, X0=None):
    M, C, K, u0, _, _, hb_nl_forces = create_motion_system() # Mxdd + Cxd + Kx = nl_forces
    samples = int(2*H+1)
    dofs = u0.shape[0] # only 1 dof for now

    # N = (H+1)*(2**2) # sampling points (needs to be power of 2)
    N = 4*H+1
    print("Sampling points:", N)
    def dynamics_residual(X):
        """
        X[:-1]: dof*(2*H+1) Fourier coefficients of the system [X0, Xc1, Xs1, ... XcH, XsH]
        where Xx is [xx_0, xx_1, ... xx_M] with M = dofs
        X[-1]: Fourier series base frequency
        """
        Om = X[-1]
        R_lin = np.zeros(X.shape[0]-1) # we exclude the continuation parameter equation

        # Compute the linear forces in Fourier domain
        R_lin[0:dofs] = K @ X[0:dofs]
        for k in range(1, H+1):
            i = (2*k-1) * dofs
            R_lin[i:i+dofs] = (K - (k*Om)**2 * M) @ X[i:i+dofs] + k * Om * C @ X[i+dofs:i+2*dofs]
            R_lin[i+dofs:i+2*dofs] = - k * Om * C @ X[i:i+dofs] + (K - (k*Om)**2 * M) @ X[i+dofs:i+2*dofs]

        # Optimized FFT version of AFT
        T = 2 * np.pi / Om      # period
        dt = T / N              # time step
        q = np.fft.irfft(X_to_complex(X[:-1].reshape(samples, dofs).T), N, axis=1, norm='forward') # no scaling
        # q_dot = np.fft.ifft(1j * w * Q_fft).real
        # q_ddot = np.fft.ifft(- (w**2) * Q_fft).real
        R_nlt = np.zeros((dofs, N))
        dt = T / N
        for s in range(N):
            t_n = s * dt
            R_nlt[:, s] = - hb_nl_forces(t_n, q[:, s], None, Om)
        
        R_nl_fft = np.fft.rfft(R_nlt, N, axis=1, norm='backward') # no scaling
        R_nl = X_to_real(R_nl_fft[:, :H+1] / N).T.reshape(-1)

        return R_lin + R_nl
    
    if X0 is None:
        # Duffing initial conditions from NLvib
        x0 = np.zeros(dofs*(2*H+1))
        q_real = -(omega_start**2) * mu + kappa
        q_imag = omega_start*zeta

        x0[1] = P*q_real / (q_real**2 + q_imag**2)
        x0[2] = P*q_imag / (q_real**2 + q_imag**2)

        # Full state initial conditions
        X0 = np.zeros(dofs*(2*H+1) + 1)
        X0[:-1] = x0
        X0[-1] = omega_start
        # print(X0)
        # plot_hb_timedomain(0.0, 100.0, 0.1, 1, X0[:-1], X0[-1], H)
    
    print("Initial guess X0:", X0)

    # Continuation framework
    max_continuation_steps = 5000
    ds_min = 1e-5
    ds_max = (omega_end - omega_start) / 5
    target_prediction_error = 1e-2 # l2 norm

    if omega_end > omega_start:
        param_direction = 1
        direction = 1
    else:
        param_direction = -1
        direction = -1

    X_ref = X0.copy()
    X_old = X0.copy()
    z_ref = np.zeros_like(X0)
    z_ref[-1] = 1

    res0 = dynamics_residual(X0)
    print("Initial spectral residual norm:", np.linalg.norm(res0))
    # print(np.concatenate(([res0[0]], res0[1::2]/2, res0[2::2]/2)))
    
    Xp, info, ier, mesg = sp.optimize.fsolve(
        extended_residual,
        X0,
        args=(X_ref, z_ref, dynamics_residual,True),
        full_output=True,
        xtol = 1e-4
    ) # initial step
    if ier != 1:
        print(f"Initial step failed: {mesg}")
        return
    
    print(Xp)
    # X0[:-1] = Xp.x[:-1] # TODO: check if we cant just copy the whole thing
    X0 = Xp.copy()

    X_mat = np.zeros((X0.shape[0], max_continuation_steps))
    X_mat[:, 0] = X0

    iteration = 1
    while iteration < max_continuation_steps:
        J = numerical_jac(lambda X: extended_residual(X, X_ref, z_ref, dynamics_residual, False), X0)

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
                args=(X_ref, z_ref, dynamics_residual,False),
                full_output=True,
                xtol = 1e-4
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

        # step control
        # e = np.linalg.norm(Xp - X0)
        # ds = max(min(ds * np.sqrt(target_prediction_error / e), ds_max), ds_min)

        print(f"omega: {X0[-1]:.3f}, ds: {ds:.2e}, nfev: {info['nfev']}")

        # history
        X_mat[:, iteration] = X0
        iteration += 1
        if (X0[-1] - omega_end) * param_direction >= 0:
            print("Continuation reached the end")
            break

    if getenv("PLOT"):
        fig = px.scatter(x=X_mat[-1, :iteration], y=np.sqrt(X_mat[1, :iteration]**2 + X_mat[2, :iteration]**2), title="Duffing Oscillator")
        fig.show()

    plot_hb_timedomain(0.0, 100.0, 0.1, dofs, X_mat[:-1, 0], X_mat[-1, 0], H).show()
    plot_hb_timedomain(0.0, 100.0, 0.1, dofs, X_mat[:-1, iteration-1], X_mat[-1, iteration-1], H).show()

if __name__ == "__main__":
    # HB Continuation
    H = 30
    # assert ((H+1) & H) == 0 # or log2(H+1) is integer
    omega_start = 0.1
    omega_end = 8
    ds = 0.02
    # Time integration
    t_final = 2000.0
    dt = 0.1

    t, u = integrate_duffing(t_final, dt, omega_start)

    # 1. Extract the last 25% of the signal
    N = len(t)
    idx_start = int(0.75 * N)
    t_tr = t[idx_start:]
    u_tr = u[0, idx_start:]
    N_tr = len(t_tr)
    # if getenv("PLOT"):
    #     fig = px.line(x=t_tr, y=u_tr, title="Duffing Oscillator Time integration")
    #     fig.show()

    w = np.hanning(N_tr)         # Create the Hann window of the same length as u_tr
    u_tr_windowed = u_tr * w     # Multiply the signal by the window

    U_fft = np.fft.fft(u_tr_windowed)
    norm_factor = np.sum(w)
    freqs = np.fft.fftfreq(N_tr, dt)

    pos = freqs > 0
    f_pos = freqs[pos]
    U_pos = U_fft[pos]
    amp = np.abs(U_pos)
    i0 = np.argmax(amp)
    f0 = f_pos[i0]
    omega0 = 2*np.pi * f0
    print("Fundamental frequency = {:.4f} Hz, ω₀ = {:.4f}".format(f0, omega0))

    a0 = np.real(U_fft[0]) / norm_factor
    a_coeffs = np.zeros(H+1)  # a0,..., aH; a0 has been computed already.
    b_coeffs = np.zeros(H+1)  # here b_coeffs[0] is not used.
    a_coeffs[0] = a0

    for h in range(1, H+1):
        target = h * f0
        # Find index among positive frequencies closest to target
        i = np.argmin(np.abs(f_pos - target))
        Y = U_pos[i]
        # Multiply by 2 as usual since you’re using a one-sided spectrum.
        a_coeffs[h] = 2 * np.real(Y) / norm_factor
        b_coeffs[h] = -2 * np.imag(Y) / norm_factor

    X0_list = [a_coeffs[0]]
    for h in range(1, H + 1):
        X0_list.append(a_coeffs[h])
        X0_list.append(b_coeffs[h])
    X0_list.append(omega0)
    X0 = np.array(X0_list)

    assert X0.shape[0] == (2*H+1)+1

    fig = plot_hb_timedomain(t_tr[0], t_tr[-1], 0.1, 1, X0[:-1], X0[-1], H)
    fig.add_trace(
        go.Scatter(
            x=t_tr,  # x values for new line
            y=u_tr,  # y values for new line
            name='Time integration',  # legend label
            line=dict(color='red')  # optional: customize line color
        )
    )
    fig.show()
    hb_duffing(H, omega_start, omega_end, ds, None)

# TODO:
# 1. Phase shifting on initial guess using angle rotations -> make initial guess more robust
# 2. Try analytical 2dof model from Krack p70 -> verify the multi-dof implementation
# 3. Duffing force continuation with set omega for harmonics -> verify that other parameter continuation works (generalize better)
# 4. Duffing force continuation with omega as unknown using integral orthogonality condition -> test if it works
# 5. Finish writing the HB-VLM force calculation in frequency domain -> prereq
# 6. Try 2dof with pitch plunge equations and HB-VLM solver coupled via files -> the real thing
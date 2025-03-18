import numpy as np
import scipy as sp
import plotly.express as px

def create_dft_matrix(omega, harmonics=3):
    unknowns = 2 * harmonics + 1
    period = 2.0 * np.pi / omega

    dft = np.zeros((unknowns, unknowns))
    ddft = np.zeros((unknowns, unknowns))
    dddft = np.zeros((unknowns, unknowns))

    sqrt_unknowns = 1 / np.sqrt(unknowns)
    sqrt_unknowns_2 = np.sqrt(2 / unknowns)
    
    for i in range(unknowns): # first col
        dft[i, 0] = sqrt_unknowns
        ddft[i, 0] = 0
        dddft[i, 0] = 0
    for j in range(1, unknowns, 2):
        k = (j + 1) / 2
        for i in range(0, unknowns):
            tn = (i / unknowns) * period
            dft[i, j] = np.cos(omega * tn * k) * sqrt_unknowns_2
            dft[i, j + 1] = np.sin(omega * tn * k) * sqrt_unknowns_2
            ddft[i, j] = - omega * k * np.sin(omega * tn * k) * sqrt_unknowns_2
            ddft[i, j + 1] = omega * k * np.cos(omega * tn * k) * sqrt_unknowns_2
            dddft[i, j] = - (omega * k)**2 * np.cos(omega * tn * k) * sqrt_unknowns_2
            dddft[i, j + 1] = - (omega * k)**2 * np.sin(omega * tn * k) * sqrt_unknowns_2

    return dft, ddft, dddft

def create_fourier_basis(omega, harmonics, t):
    unknowns = 2 * harmonics + 1
    basis = np.zeros((unknowns))
    dbasis = np.zeros((unknowns))
    ddbasis = np.zeros((unknowns))
    sqrt_unknowns = 1 / np.sqrt(unknowns)
    sqrt_unknowns_2 = np.sqrt(2 / unknowns)
    basis[0] = sqrt_unknowns
    dbasis[0] = 0
    ddbasis[0] = 0
    for i in range(harmonics):
        k = float(i+1)
        basis[2 * i + 1] = sqrt_unknowns_2 * np.cos(omega * t * k)
        basis[2 * i + 2] = sqrt_unknowns_2 * np.sin(omega * t * k)
        dbasis[2 * i + 1] = - sqrt_unknowns_2 * omega * k * np.sin(omega * t * k)
        dbasis[2 * i + 2] = sqrt_unknowns_2 * omega * k * np.cos(omega * t * k)
        ddbasis[2 * i + 1] = - sqrt_unknowns_2 * (omega * k)**2 * np.cos(omega * t * k)
        ddbasis[2 * i + 2] = - sqrt_unknowns_2 * (omega * k)**2 * np.sin(omega * t * k)

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
P = 0.18
mu = 1
zeta = 0.05
kappa = 1
gamma = 0.1

# Wikipedia params
# P = 0.29
# mu = 1
# zeta = 0.3
# kappa = -1.0
# gamma = 1.0

def create_motion_system():
    # NLvib params
    omega = 0.1
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

def integrate_duffing(t_final, dt):
    M, C, K, u0, v0, nonlinear_func, _ = create_motion_system()
    t, u, v, a = nonlinear_newmark_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt)
    # t, u, v, a = nonlinear_newmark_anderson_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt)

    fig = px.line(x=t, y=u[0, :], title="Duffing Oscillator")
    fig.show()

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
#             break
    
#     # 6. Normalize tangent vector
#     z = ztmp / np.linalg.norm(ztmp) # length 1 vector
#     print(z)
#     return z

# def tangent_predictor(J, zref, Xref):
#     Q, R = np.linalg.qr(J.T)
#     z = Q[:, -1]
#     return z / np.linalg.norm(z)

def tangent_predictor(J, zref, Xref):
    _, _, vt = np.linalg.svd(J[:-1, :], full_matrices=True)
    z = vt[-1, :]
    z = z / np.linalg.norm(z)
    print(z)
    return z

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

def plot_hb_timedomain(t_final, dt, dofs, X, omega, harmonics):
    # Plot the result in time domain
    samples = 2*harmonics+1
    vec_t = np.arange(0, t_final + dt, dt)
    sol = np.zeros((3*dofs, vec_t.shape[0])) # u, v, a
    uf_sol_ = X.reshape(samples, dofs).T
    # uf_sol_ = xf0.reshape(samples, dofs).T
    for i, t in enumerate(vec_t):
        b, db, ddb = create_fourier_basis(omega, harmonics, t)
        sol[0:dofs, i] = uf_sol_ @ b
        sol[dofs:2*dofs, i] = uf_sol_ @ db
        sol[2*dofs:3*dofs, i] = uf_sol_ @ ddb

    fig = px.line(x=vec_t, y=sol[0, :], title=f"Duffing Oscillator (omega={omega})")

    fig.show()

def hb_duffing(harmonics, omega_start, omega_end, ds=.01):
    M, C, K, u0, v0, _, hb_nl_forces = create_motion_system()
    samples = 2*harmonics+1
    dofs = u0.shape[0] # only 1 dof for now

    # Extended residual function
    # TODO: separate the continuation part
    def f(X, X_ref, z_ref, init: bool): # Xf is the full state vector
        uf = X[:-1].reshape(samples, dofs).T # Matrix where (i, j) is the j-th fourier coeff for i-th dof
        residual = np.zeros_like(X)
        R = np.zeros_like(uf)
        omega = X[-1]
        d, dd, ddd = create_dft_matrix(omega, harmonics)

        # pos, vel and accel at time samples using IDFT
        u = uf @ d.T
        v = uf @ dd.T
        a = uf @ ddd.T

        # Alternating frequency-time scheme
        for s in range(samples):
            period = 2.0 * np.pi / omega
            t_n = (s / samples) * period
            R[:, s] = M @ a[:, s] + C @ v[:, s] + K @ u[:, s] - hb_nl_forces(t_n, u[:, s], v[:, s], omega)

        R = R @ d # DFT
        residual[:-1] = R.T.reshape(-1)

        if init: # local parametrization
            residual[-1] = np.dot(z_ref, X - X_ref)
        else: # arc-length parametrization
            residual[-1] = np.dot(X - X_ref, X - X_ref) - ds**2 # iteration on a normal plane, perpendicular to tangent

        return residual
    
    # Duffing initial conditions from NLvib
    x0 = np.zeros(dofs*(2*harmonics+1))
    q_real = -(omega_start**2) * mu + kappa
    q_imag = omega_start*zeta
    x0[1] = P*q_real / (q_real**2 + q_imag**2)
    x0[2] = P*q_imag / (q_real**2 + q_imag**2)

    # Full state initial conditions
    X0 = np.zeros(dofs*(2*harmonics+1) + 1)
    X0[:-1] = x0
    X0[-1] = omega_start
    
    # Continuation framework
    max_continuation_steps = 2000
    if omega_end > omega_start:
        direction = 1
    else:
        direction = -1

    X_ref = X0.copy()
    X_old = X0.copy()
    z_ref = np.zeros_like(X0)
    z_ref[-1] = 1

    Xp = sp.optimize.root(f, X0, args=(X_ref, z_ref, True)) # initial step
    # X0[:-1] = Xp.x[:-1] # TODO: check if we cant just copy the whole thing
    X0 = Xp.x.copy()

    X_mat = np.zeros((X0.shape[0], max_continuation_steps))
    X_mat[:, 0] = X0

    iteration = 1
    while iteration < max_continuation_steps:
        print(X0[-1])
        J = numerical_jac(lambda X: f(X, X_ref, z_ref, False), X0)

        z = tangent_predictor(J, z_ref, X_ref)

        # Take a step in the tangent direction ensuring to stay along the solution path
        if (iteration > 1) and np.dot(X0-X_old, ds*z) < 0:
            Xp = X0 - direction*ds*z
        else:
            Xp = X0 + direction*ds*z

        # Parametrizaton params
        X_ref = X0.copy()
        z_ref = z.copy()
        Xtmp = sp.optimize.root(f, Xp, args=(X_ref, z_ref, False))
        X_old = X0.copy()
        X0 = Xtmp.x.copy()
        X_mat[:, iteration] = X0
        iteration += 1
        if (X0[-1] - omega_end) * direction >= 0:
            print("Continuation reached the end")
            break
    
    fig = px.line(x=X_mat[-1, :iteration], y=X_mat[1, :iteration]**2 + X_mat[2, :iteration]**2, title="Duffing Oscillator")
    fig.show()

    plot_hb_timedomain(100.0, 0.1, dofs, X_mat[:-1, 3], X_mat[-1, 3], harmonics)

if __name__ == "__main__":
    # params
    # harmonics = 10
    # omega0 = 0.3
    # dt = 0.01
    # # dependent vars
    # unknowns = 2 * harmonics + 1
    # period = 2.0 * np.pi / omega0
    # t_start = 3*period
    t, u = integrate_duffing(300.0, 0.1)
    # samples = np.zeros((u.shape[0], unknowns))
    # xf0 = np.zeros_like(samples)

    # # sampling
    # for i in range(0, unknowns):
    #     tn = t_start + (i / unknowns) * period
    #     samples[:, i] = u[:, int(round(tn / dt))]
    #     print(f"tn: {tn}")
    #     print(f"t[int(tn / dt)]: {t[int(round(tn / dt))]}")

    # dft, _, _ = create_dft_matrix(omega0, harmonics)
    # xf0 = samples @ dft

    # hb_duffing(harmonics, xf0.T.reshape(-1), omega0)
    harmonics = 3
    omega_start = 0.5
    omega_end = 1.6
    ds = 0.05
    hb_duffing(harmonics, omega_start, omega_end, ds)

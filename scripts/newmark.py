import numpy as np

def newmark_beta(M, C, K, x0, v0, F, t, beta=1/4, gamma=1/2):
    """
    Implicit Newmark-Beta Method for Structural Dynamics.

    Parameters:
    - M, C, K: Mass, Damping, and Stiffness matrices (n x n).
    - x0: Initial displacement vector (n,).
    - v0: Initial velocity vector (n,).
    - F: External force matrix (nt x n).
    - t: Time array (nt,).
    - beta, gamma: Newmark parameters.

    Returns:
    - u: Displacement matrix (nt x n).
    - v: Velocity matrix (nt x n).
    - a: Acceleration matrix (nt x n).
    """
    n = M.shape[0]
    nt = len(t)
    dt = t[1] - t[0]

    # Initialize arrays
    u = np.zeros((nt, n))
    v = np.zeros((nt, n))
    a = np.zeros((nt, n))

    # Set initial conditions
    u[0] = x0
    v[0] = v0
    a[0] = np.linalg.solve(M, F[0] - C @ v0 - K @ x0)

    # Precompute constants
    a0 = 1 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = (gamma / beta) - 1
    a5 = (dt / 2) * (gamma / beta - 2)

    # Effective stiffness matrix
    K_eff = K + a0 * M + a1 * C
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(nt - 1):
        # Predictors
        F_eff = F[i+1] + M @ (a0 * u[i] + a2 * v[i] + a3 * a[i]) + C @ (a1 * u[i] + a4 * v[i] + a5 * a[i])

        # Solve for displacement at next time step
        u[i+1] = K_eff_inv @ F_eff

        # Compute acceleration and velocity
        a[i+1] = a0 * (u[i+1] - u[i]) - a2 * v[i] - a3 * a[i]
        v[i+1] = v[i] + gamma * dt * a[i] + gamma * dt * a[i+1]

    return u, v, a

def newmark_beta_v2(M, C, K, x0, v0, F, t, beta=1/4, gamma=1/2):
    """
    Implicit Newmark-Beta Method for Structural Dynamics.

    Parameters:
    - M, C, K: Mass, Damping, and Stiffness matrices (n x n).
    - x0: Initial displacement vector (n,).
    - v0: Initial velocity vector (n,).
    - F: External force matrix (nt x n).
    - t: Time array (nt,).
    - beta, gamma: Newmark parameters.

    Returns:
    - u: Displacement matrix (nt x n).
    - v: Velocity matrix (nt x n).
    - a: Acceleration matrix (nt x n).
    """
    n = M.shape[0]
    nt = len(t)
    dt = t[1] - t[0]

    # Initialize arrays
    u = np.zeros((nt, n))
    v = np.zeros((nt, n))
    a = np.zeros((nt, n))

    # Set initial conditions
    u[0] = x0
    v[0] = v0
    a[0] = np.linalg.solve(M, F[0] - C @ v0 - K @ x0)

    # Precompute constants
    x2 = 1
    x1 = gamma / (beta * dt)
    x0 = 1 / (beta * dt**2)
    xd0 = 1 / (beta * dt)
    xd1 = gamma / beta
    xdd0 = 1/(2*beta)
    xdd1 = - dt * (1 - gamma / (2*beta))

    # Effective stiffness matrix
    K_eff = x0 * M + x1 * C + x2 * K
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(nt - 1):
        # Predictors
        F_eff = (F[i+1]-F[i]) + M @ (xd0 * v[i] + xdd0 * a[i]) + C @ (xd1 * v[i] + xdd1 * a[i])
        # Solve for displacement at next time step
        du = K_eff_inv @ F_eff

        u[i+1] = u[i] + du
        v[i+1] = v[i] + x1 * du - xd1 * v[i] - xdd1 * a[i]
        a[i+1] = a[i] + x0 * du - xd0 * v[i] - xdd0 * a[i]

    return u, v, a

def hht_alpha(M, C, K, x0, v0, F, t, alpha=0.05):
    """
    HHT-alpha Method for Structural Dynamics.

    Parameters:
    - M, C, K: Mass, Damping, and Stiffness matrices (n x n).
    - x0: Initial displacement vector (n,).
    - v0: Initial velocity vector (n,).
    - F: External force matrix (nt x n).
    - t: Time array (nt,).
    - alpha: damping parameter (decreases response at frequencies above 1/(2*dt))

    Returns:
    - u: Displacement matrix (nt x n).
    - v: Velocity matrix (nt x n).
    - a: Acceleration matrix (nt x n).
    """
    assert alpha >= 0, "alpha must be non-negative"
    assert alpha <= 1/3
    
    n = M.shape[0]
    nt = len(t)
    dt = t[1] - t[0]

    # Initialize arrays
    u = np.zeros((nt, n))
    v = np.zeros((nt, n))
    a = np.zeros((nt, n))

    # Set initial conditions
    u[0] = x0
    v[0] = v0
    a[0] = np.linalg.solve(M, F[0] - C @ v0 - K @ x0)

    beta = (1+alpha**2)/4
    gamma = 1/2 + alpha
    # Precompute constants
    a0 = 1 - alpha
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt**2)
    a3 = 1 / (beta * dt)
    a4 = gamma / beta
    a5 = 1/(2*beta)
    a6 = dt * (1 - gamma / (2*beta))

    # Effective stiffness matrix
    K_eff = a0 * K + a2 * M + a0 * a1 * C
    K_eff_inv = np.linalg.inv(K_eff)
    for i in range(nt - 1):
        # Predictors
        F_eff = M @ (a3 * v[i] + a5 * a[i]) + C @ (a0 * a4 * v[i] - a0 * a6 * a[i] - alpha * a1 * (u[i] - u[i-1]) + alpha * a4 * v[i-1] - alpha * a6 * a[i-1]) + K @ (-alpha * (u[i] - u[i-1])) + a0 * (F[i+1]-F[i]) + alpha * (F[i] - F[i-1])
        # displacement increment
        du = K_eff_inv @ F_eff

        u[i+1] = u[i] + du
        v[i+1] = v[i] + a1 * du - a4 * v[i] + a6 * a[i]
        a[i+1] = a[i] + a2 * du - a3 * v[i] - a5 * a[i]

    return u, v, a

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define system matrices
    K = np.array([[400, -200, 0],
                [-200, 400, -200],
                [0, -200, 200]], dtype=float)

    C = np.array([[0.55, -0.2, 0],
                [-0.2, 0.4, -0.2],
                [0, -0.2, 0.35]], dtype=float)

    M = np.eye(3, dtype=float)

    # Initial conditions
    x0 = np.array([0, 0, 0], dtype=float)
    v0 = np.array([1, 1, 1], dtype=float)

    plt.figure(figsize=(10, 6))

    dts = [0.1, 0.05, 0.01]
    total_time = 20

    for dt in dts:
        t = np.arange(0, total_time + dt, dt)

        # External force (zero for this example)
        F = np.zeros((len(t), 3), dtype=float)

        # Perform integration
        # u, v, a = hht_alpha(M, C, K, x0, v0, F, t)
        u, v, a = newmark_beta_v2(M, C, K, x0, v0, F, t)

        # Plot acceleration of the first degree of freedom
        plt.plot(t, a[:, 0], label=f"HHT-alpha (dt={dt})")

    with open("build/windows/x64/debug/newmark_3dof_cuda.txt", "r") as f:
        first_line = f.readline()
        dof, tsteps = map(int, first_line.split())
        result_cpp = np.zeros((dof * 3 + 1, tsteps))
        for step, line in enumerate(f):
            result_cpp[:, step] = np.array(list(map(float, line.split())))

    plt.plot(result_cpp[0, :], result_cpp[7, :], "--",label="C++ (CPU)")

    plt.title('Acceleration Response of the First Degree of Freedom')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (units/s²)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
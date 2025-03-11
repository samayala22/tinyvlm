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

    # assert np.allclose(np.linalg.inv(dft), dft.T)
    # assert np.allclose(np.linalg.inv(ddft), ddft.T)
    # assert np.allclose(np.linalg.inv(dddft), dddft.T)

    return dft, ddft, dddft

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
    t_steps = int((t_final + dt)/ dt)
    n = u0.shape[0] # number of equations
    vec_t = np.arange(0, t_final + dt, dt)
    u = np.zeros((n, t_steps+1))
    v = np.zeros((n, t_steps+1))
    a = np.zeros((n, t_steps+1))
    f_curr = nonlinear_func(0.0, u0, v0)
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = np.linalg.solve(M, f_curr - C @ v0 - K @ u0)

    avg_iters = 0

    for i in range(0, t_steps):
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

def create_duffing_system(delta = 0.3, alpha = -1.0, beta = 1.0, gamma = 0.37, omega = 1.2):
    def nonlinear_func(t, u, v):
        return np.array([gamma * np.cos(omega * t) - beta * u[0]**3])
    
    M = np.array([[1.0]])
    C = np.array([[delta]])
    K = np.array([[alpha]])

    u0 = np.array([1.0])
    v0 = np.array([0.0])

    return M, C, K, u0, v0, nonlinear_func

def integrate_duffing():
    M, C, K, u0, v0, nonlinear_func = create_duffing_system()
    t_final = 150.0
    dt = 0.2
    t, u, v, a = nonlinear_newmark_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt)
    # t, u, v, a = nonlinear_newmark_anderson_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt)

    fig = px.line(x=t, y=u[0, :], title="Duffing Oscillator")
    fig.show()

def hb_duffing(harmonics=3):
    M, C, K, u0, v0, nonlinear_func = create_duffing_system()
    samples = 2*harmonics+1
    dofs = u0.shape[0] # only 1 dof for now
    omega0 = 1.0

    def f(xf):
        omega_k = xf[-1]
        uf = xf[:-1].reshape(samples, dofs).T # Matrix where each col is the fourier coeffs for row idx dof
        R = np.zeros_like(xf)
        d, dd, ddd = create_dft_matrix(omega_k, harmonics)

        # pos, vel and accel at time samples
        u = uf @ d.T
        v = uf @ dd.T
        a = uf @ ddd.T

        for s in range(samples):
            period = 2.0 * np.pi / omega_k
            t_n = (s / samples) * period
            R[dofs*s:dofs*(s+1)] = M @ a[:, s] + C @ v[:, s] + K @ u[:, s] - nonlinear_func(t_n, u[:, s], v[:, s])

        # TODO: couple back with a DFT
        R[-1] = omega0 - omega_k
        return 

    xf0 = np.zeros(dofs*(2*harmonics+1)+1)
    xf0[-1] = omega0

    xf_sol = sp.optimize.newton_krylov(f, xf0)

if __name__ == "__main__":
    integrate_duffing()
    # hb_duffing()

import numpy as np
from tqdm import tqdm

def newmark_beta_step(M, C, K, v_i, a_i, delta_F, dt, beta=1/4, gamma=1/2):
    """
    Implicit Newmark-Beta Method for Structural Dynamics.

    Parameters:
    - M, C, K: Mass, Damping, and Stiffness matrices (n x n).
    - beta, gamma: Newmark parameters.

    Returns: incremental variation for displacement, velocity, and acceleration.
    """

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
    F_eff = delta_F + M @ (xd0 * v_i + xdd0 * a_i) + C @ (xd1 * v_i + xdd1 * a_i)
    du = np.linalg.solve(K_eff, F_eff)
    dv = x1 * du - xd1 * v_i - xdd1 * a_i
    da = x0 * du - xd0 * v_i - xdd0 * a_i

    return du, dv, da

def nonlinear_newmark_solve(M, C, K, u0, v0, nonlinear_func, t_final, dt, tol=1e-8, max_iter=100):
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

    for i in tqdm(range(0, t_steps-1)):
        t = i*dt
        f_next = f_curr.copy()
        du = np.zeros(n)
        du_k = np.zeros(n) + 1
        iteration = 0
        while (np.linalg.norm(du_k - du) / len(du) > tol) and (iteration < max_iter):
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

def rk2_step(f: callable, x, t, dt, i):
    k1 = f(t, x[i])
    x_mid = x[i] + k1 * (dt / 2)
    t_mid = t + dt / 2
    k2 = f(t_mid, x_mid)
    x[i+1] = x[i] + 0.5 * (k1 + k2) * dt
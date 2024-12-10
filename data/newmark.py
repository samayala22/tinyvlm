import numpy as np
import matplotlib.pyplot as plt

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
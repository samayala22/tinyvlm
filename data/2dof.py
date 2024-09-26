import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.special as scsp
from tqdm import tqdm

EPS_sqrt_f = np.sqrt(1.19209e-07)

def newmark_beta_step(M, C, K, u_i, v_i, a_i, F_i, F_ip1, dt, beta=1/4, gamma=1/2):
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
    F_eff = (F_ip1-F_i) + M @ (xd0 * v_i + xdd0 * a_i) + C @ (xd1 * v_i + xdd1 * a_i)
    du = np.linalg.solve(K_eff, F_eff)
    dv = x1 * du - xd1 * v_i - xdd1 * a_i
    da = x0 * du - xd0 * v_i - xdd0 * a_i

    return du, dv, da

def rk2_step(f: callable, x, t, dt, i):
    k1 = f(t, x[i])
    x_mid = x[i] + k1 * (dt / 2)
    t_mid = t + dt / 2
    k2 = f(t_mid, x_mid)
    x[i+1] = x[i] + 0.5 * (k1 + k2) * dt

def theo_fun(k):
    H1 = scsp.hankel2(1, k)
    H0 = scsp.hankel2(0, k)
    C = H1 / (H1 + 1.0j * H0)
    return C

# Some info in Katz Plotkin p414 (eq 13.73a)
# Jone's approximation of Wagner function
b0 = 1
b1 = -0.165
b2 = -0.335
beta_1 = 0.0455
beta_2 = 0.3

# UVLM parameters
rho = 1 # fluid density
u_inf = 1 # freestream
ar = 10000 # aspect ratio
b = 0.5 # half chord
c = 2*b # chord
a = ar / c # full span
pitch_axis = -0.5
pa = pitch_axis / b

# Theodorsen numerical simulation param
dt = 0.01
t_final = 5 # 30s
vec_t = np.arange(0, t_final, dt)

# Aeroelastic model
m = 2.0 # mass
S_a = 0.2 # first moment of inertia
I_a = 4.0 # second moment of inertia
K_h = 800.0 # linear stiffness
K_a = 150.0 # torsional stiffness

M = np.array([[m, S_a],[S_a, I_a]])
C = np.zeros((2,2))
K = np.array([[K_h, 0],[0, K_a]])

u = np.zeros((2, len(vec_t)))
v = np.zeros((2, len(vec_t)))
a = np.zeros((2, len(vec_t)))
F = np.zeros((2, len(vec_t)))

# augmented aero states
vec_x1 = np.zeros(len(vec_t))
vec_x2 = np.zeros(len(vec_t))

# Initial condition
u[:, 0] = np.array([-0.5, 0])
v[:, 0] = np.array([0, 0])
a[:, 0] = np.linalg.solve(M, F[:, 0] - C @ v[:,0] - K @ u[:,0])

fig, axs = plt.subplot_mosaic(
    [["CL"], ["h"], ["alpha"]],  # Disposition des graphiques
    constrained_layout=True,  # Demander à Matplotlib d'essayer d'optimiser la disposition des graphiques pour que les axes ne se superposent pas
    figsize=(11, 8),  # Ajuster la taille de la figure (x,y)
)

def w(s: float):
    idx = int(s / dt)
    return u_inf * u[1, idx] + v[0,idx] + b * (0.5 - pa) * v[1, idx]

def dx1ds(s: float, x1: float): return (u_inf / b) * (b1 * beta_1 * w(s) - beta_1 * x1)
def dx2ds(s: float, x2: float): return (u_inf / b) * (b2 * beta_2 * w(s) - beta_2 * x2)

def aero(i):
    t = vec_t[i]

    rk2_step(dx1ds, vec_x1, vec_t[i-1], dt, i-1)
    rk2_step(dx2ds, vec_x2, vec_t[i-1], dt, i-1)
    x1 = vec_x1[i]
    x2 = vec_x2[i]

    L_m = rho * b * b * np.pi * (u_inf * v[1,i] + a[0,i] - b * pa * a[1,i])
    L_c = -2 * np.pi * rho * u_inf * b * (-(b0 + b1 + b2) * w(t) + x1 + x2)
    L = L_m + L_c
    M = (0.25 + pitch_axis) * np.cos(u[1,i]) * L
    F[0,i] = -L
    F[1,i] = M
    # C_l = L / (0.5 * rho * u_inf * u_inf * c)

def central_difference_step(M, C, K, u_prev, u, F_i, dt):
    M_inv = np.linalg.inv(M)
    a = M_inv @ (F_i - C @ ((u - u_prev) / dt) - K @ u)
    u_next = 2 * u - u_prev + dt**2 * a
    return u_next

# Initialize u_prev
u_prev = u[:,0] - dt * v[:,0] + 0.5 * dt**2 * a[:,0]
for i in tqdm(range(1, len(vec_t))):
    F_i = F[:,i-1]
    u[:,i] = central_difference_step(M, C, K, u_prev, u[:,i-1], F_i, dt)
    v[:,i] = (u[:,i] - u_prev) / (2 * dt)
    a[:,i] = (u[:,i] - 2 * u[:,i-1] + u_prev) / dt**2
    u_prev = u[:,i-1]
    aero(i)

# # Newmark
# for i in tqdm(range(len(vec_t)-1)):
#     t = vec_t[i]

#     du, dv, da = newmark_beta_step(M, C, K, u[:,i], v[:,i], a[:,i], F[:,i], F[:,i+1], dt)
#     u[:,i+1] = u[:,i] + du
#     v[:,i+1] = v[:,i] + dv
#     a[:,i+1] = a[:,i] + da
#     if (i != 0):
#         aero(i)

#     # LOOSE COUPLING
#     aero(i+1)
#     du, dv, da = newmark_beta_step(M, C, K, u[:,i], v[:,i], a[:,i], F[:,i], F[:,i+1], dt)
#     u[:,i+1] = u[:,i] + du
#     v[:,i+1] = v[:,i] + dv
#     a[:,i+1] = a[:,i] + da
    
#     # ITERATIVE COUPLING
#     # du_k = np.zeros(2)
#     # iteration = 0
#     # while (np.linalg.norm(du_k - du) / 2 > 1e-4):
#     #     du_k = du[:]
#     #     aero(i+1)
#     #     du, dv, da = newmark_beta_step(M, C, K, u[:,i], v[:,i], a[:,i], F[:,i], F[:,i+1], dt)
#     #     u[:,i+1] = u[:,i] + du
#     #     v[:,i+1] = v[:,i] + dv
#     #     a[:,i+1] = a[:,i] + da
#     #     iteration += 1
#     # print("iters: ", iteration)

axs["h"].plot(vec_t, u[0, :])
axs["alpha"].plot(vec_t, u[1, :])

axs["h"].set_xlabel('t')
axs["h"].set_ylabel('h')
axs["h"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')

axs["alpha"].set_xlabel('t')
axs["alpha"].set_ylabel('alpha')
axs["alpha"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')

# plt.suptitle("Verification of UVLM with Theodorsen harmonic heave motion")

plt.show()

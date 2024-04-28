import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Two point central difference
def derivative(f, x):
    EPS_sqrt_f = np.sqrt(1.19209e-07)
    return (f(x + EPS_sqrt_f) - f(x - EPS_sqrt_f)) / (2 * EPS_sqrt_f)

def derivative2(f, x):
    EPS_sqrt_f = np.sqrt(1.19209e-07)
    return (f(x + EPS_sqrt_f) - 2 * f(x) + f(x - EPS_sqrt_f)) / (EPS_sqrt_f ** 2)

def solve_ivp(x0: float, s0: float, sf: float, f: callable):
    return spi.solve_ivp(f, [s0, sf], [x0]).y[-1] # return only the result at t=sf

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
ar = 500 # aspect ratio
b = 0.5 # half chord
c = 2*b # chord
a = ar / c # full span
pitch_axis = -1 # leading edge

def atime(t: float): return 2. * u_inf * t / c

amplitudes = [0.1] 
reduced_frequencies = [1.0]

t_final = 10
nb_pts = 500
vec_t = np.linspace(0, t_final, nb_pts)

fig, axs = plt.subplot_mosaic(
    [["time"],["error"], ["heave"]],  # Disposition des graphiques
    constrained_layout=True,  # Demander à Matplotlib d'essayer d'optimiser la disposition des graphiques pour que les axes ne se superposent pas
    figsize=(11, 6),  # Ajuster la taille de la figure (x,y)
)

for amp, k in zip(amplitudes, reduced_frequencies):
    # heave parameters
    amplitude = amp / c
    omega = k * u_inf / c # pitch frequency

    # sudden acceleration
    def pitch(t): return np.radians(5)
    def heave(t): return 0

    # pure heaving
    # def pitch(t): return 0
    # def heave(t): return -amplitude * np.sin(omega * t)

    # pure pitching
    # def pitch(t): return np.radians(np.sin(omega * t))
    # def heave(t): return 0

    def w(s: float): 
        return u_inf * pitch(s) + derivative(heave, s) + b * (0.5 - pitch_axis) * derivative(pitch, s)

    def dx1ds(s: float, x1: float): return b1 * beta_1 * w(s) - beta_1 * x1
    def dx2ds(s: float, x2: float): return b2 * beta_2 * w(s) - beta_2 * x2

    x1_solution = spi.solve_ivp(dx1ds, [0, t_final], [0], t_eval=vec_t)
    x2_solution = spi.solve_ivp(dx2ds, [0, t_final], [0], t_eval=vec_t)
    
    def x1(s: float): return np.interp(s, x1_solution.t, x1_solution.y[0])
    def x2(s: float): return np.interp(s, x2_solution.t, x2_solution.y[0])

    def cl_theodorsen(t: float): # using Wagner functions and Kholodar formulation
        L_m = rho * b * b * np.pi * (u_inf * derivative(pitch, t) + derivative2(heave, t) - b * pitch_axis * derivative2(pitch, t))
        L_c = -2 * np.pi * rho * u_inf * b * (-(b0 + b1 + b2) * w(t) + x1(t) + x2(t))
        return (L_m + L_c) / (0.5 * rho * u_inf * u_inf * c)
    
    cl_theo = np.array([cl_theodorsen(ti) for ti in vec_t])
    coord_z = np.array([-heave(ti) / c for ti in vec_t])
    angle = np.array([np.degrees(pitch(ti)) for ti in vec_t])

    axs["time"].plot(vec_t, cl_theo, label=f"k={k}")
    axs["heave"].plot(angle[int(nb_pts//2):], cl_theo[int(nb_pts//2):], label=f"k={k}")

uvlm_cl = []
uvlm_t = []
uvlm_z = []
uvlm_alpha = []
with open("build/windows/x64/debug/cl_data.txt", "r") as f:
    for line in f:
        time_step, z, cl, alpha = line.split()
        uvlm_t.append(float(time_step))
        uvlm_z.append(float(z))
        uvlm_cl.append(float(cl))
        uvlm_alpha.append(float(alpha))

n = len(uvlm_t) # number of time steps
uvlm_alpha = np.array(uvlm_alpha)
uvlm_cl = np.array(uvlm_cl)
analytical_cl = np.array([np.interp(ut, vec_t, cl_theo) for ut in uvlm_t])
err_rel = np.abs((uvlm_cl-analytical_cl)/analytical_cl)
print("Avg rel error: ", 100.0 * np.mean(err_rel))

axs["error"].plot(uvlm_t, 100.0 * err_rel, label="rel (%)")
axs["time"].scatter(uvlm_t, uvlm_cl, label="UVLM", facecolors='none', edgecolors='b', s=15)
# axs["heave"].scatter(uvlm_z[len(uvlm_cl)//4:], uvlm_cl[len(uvlm_cl)//4:], label="UVLM", facecolors='none', edgecolors='b', s=15)
# axs["heave"].scatter(np.degrees(uvlm_alpha[len(uvlm_cl)//4:]), uvlm_cl[len(uvlm_cl)//4:], label="UVLM", facecolors='none', edgecolors='b', s=15)
axs["heave"].scatter(uvlm_alpha[n//2:], uvlm_cl[n//2:], label="UVLM", facecolors='none', edgecolors='b', s=15)

# axs["time"].plot(vec_t, [0.548311] * len(vec_t), label="VLM (alpha=5)")

axs["time"].set_xlabel('t')
axs["time"].set_ylabel('CL')
axs["time"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
axs["time"].legend()

axs["error"].set_xlabel('t')
axs["error"].set_ylabel('err')
axs["error"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
axs["error"].legend()

axs["heave"].set_xlabel('h/c')
axs["heave"].set_ylabel('CL')
axs["heave"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
axs["heave"].legend()

plt.suptitle("Verification of UVLM with Theodorsen harmonic heave motion")
# plt.suptitle("Verification of UVLM with sudden acceleration motion")

plt.show()

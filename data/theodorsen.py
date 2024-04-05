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

# Jone's approximation of Wagner function
b0 = 1
b1 = -0.165
b2 = -0.335
beta_1 = 0.0455
beta_2 = 0.3

# UVLM parameters
rho = 1 # fluid density
u_inf = 1 # freestream
b = 0.5 # half chord
a = 10 # full span

amplitudes = [0.1, 0.1, 0.1] 
reduced_frequencies = [0.5, 0.75, 1.5]

t_final = 30
t = np.linspace(0, t_final, 500)

fig, axs = plt.subplot_mosaic(
    [["time"],["heave"]],  # Disposition des graphiques
    constrained_layout=True,  # Demander à Matplotlib d'essayer d'optimiser la disposition des graphiques pour que les axes ne se superposent pas
    figsize=(16, 9),  # Ajuster la taille de la figure (x,y)
)

for amp, k in zip(amplitudes, reduced_frequencies):
    # heave parameters
    amplitude = amp / (2*b)
    omega = k * u_inf / (2*b) # pitch frequency

    def pitch(t): return 0
    def heave(t): return amplitude * np.sin(omega * t)

    def w(s: float): return u_inf * pitch(s) + derivative(heave, s) + b * (0.5 - a) * derivative(pitch, s)

    def dx1ds(s: float, x1: float): return b1 * beta_1 * w(s) - beta_1 * x1
    def dx2ds(s: float, x2: float): return b2 * beta_2 * w(s) - beta_2 * x2

    def cl_theodorsen(t: float):
        L_m = rho * b * b * np.pi * (u_inf * derivative(pitch, t) + derivative2(heave, t) - b * a * derivative2(pitch, t))
        L_c = -2 * np.pi * rho * u_inf * b * ((b0 + b1 + b2) * w(t) + solve_ivp(0, 0, t, dx1ds)[-1] + solve_ivp(0, 0, t, dx2ds)[-1])
        return (L_m + L_c) / (0.5 * rho * u_inf * u_inf * a * b)

    cl = np.array([cl_theodorsen(ti) for ti in t])
    coord_z = np.array([heave(ti) / (2*b) for ti in t])

    axs["time"].plot(t, cl, label=f"k={k}")
    axs["heave"].plot(coord_z[len(cl)//2:], cl[len(cl)//2:], label=f"k={k}")

axs["time"].set_xlabel('t')
axs["time"].set_ylabel('CL')
axs["time"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
axs["time"].legend()

axs["heave"].set_xlabel('h/c')
axs["heave"].set_ylabel('CL')
axs["heave"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
axs["heave"].legend()

plt.show()

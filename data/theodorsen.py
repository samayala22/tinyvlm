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

def pade_approximation(k: float):
    return (0.5177*k*k + 0.2752*k + 0.01576) / (k*k + 0.3414*k + 0.01582)

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

amplitudes = [0.1, 0.1, 0.1] 
reduced_frequencies = [0.5, 0.75, 1.5]

t_final = 30
t = np.linspace(0, t_final, 500)

fig, axs = plt.subplot_mosaic(
    [["time"],["heave"]],  # Disposition des graphiques
    constrained_layout=True,  # Demander à Matplotlib d'essayer d'optimiser la disposition des graphiques pour que les axes ne se superposent pas
    figsize=(11, 6),  # Ajuster la taille de la figure (x,y)
)

for amp, k in zip(amplitudes, reduced_frequencies):
    # heave parameters
    amplitude = amp / (2*b)
    omega = k * u_inf / (2*b) # pitch frequency

    # sudden acceleration
    def pitch(t): return np.radians(5)
    def heave(t): return 0

    # pure heaving
    # def pitch(t): return 0
    # def heave(t): return amplitude * np.sin(omega * t)
    
    def w(s: float): 
        return u_inf * pitch(s) + derivative(heave, s) + b * (0.5 - pitch_axis) * derivative(pitch, s)

    def dx1ds(s: float, x1: float): return b1 * beta_1 * w(s) - beta_1 * x1
    def dx2ds(s: float, x2: float): return b2 * beta_2 * w(s) - beta_2 * x2

    x1_solution = spi.solve_ivp(dx1ds, [0, t_final], [0], t_eval=t)
    x2_solution = spi.solve_ivp(dx2ds, [0, t_final], [0], t_eval=t)
    
    def x1(s: float): return np.interp(s, x1_solution.t, x1_solution.y[0])
    def x2(s: float): return np.interp(s, x2_solution.t, x2_solution.y[0])

    def cl_theodorsen(t: float): # using Wagner functions and Kholodar formulation
        L_m = rho * b * b * np.pi * (u_inf * derivative(pitch, t) + derivative2(heave, t) - b * pitch_axis * derivative2(pitch, t))
        L_c = -2 * np.pi * rho * u_inf * b * ((b0 + b1 + b2) * w(t) + x1(t) + x2(t))
        return (L_m + L_c) / (0.5 * rho * u_inf * u_inf * c)

    # def cl_theodorsen(t: float): # using Pade approximation
        # return 0.5 * np.pi * (derivative2(heave, t) + derivative(pitch, t) - 0.5 * pitch_axis * derivative2(pitch, t)) + 2.0 * np.pi * (pitch(t) + derivative(heave, t) + 0.5 * derivative(pitch, t) * (0.5 - pitch_axis)) * pade_approximation(k)
        # L = rho * b * b * np.pi * (u_inf * derivative(pitch, t) + derivative2(heave, t) - b * pitch_axis * derivative2(pitch, t)) + 2 * np.pi * rho * u_inf * b * (u_inf * pitch(t) + derivative(heave, t) + b * (0.5 - pitch_axis) * derivative(pitch, t)) * pade_approximation(k)
        # return L / (0.5 * rho * u_inf * u_inf * a * (2*b))
    
    cl = np.array([cl_theodorsen(ti) for ti in t])
    coord_z = np.array([heave(ti) / (2*b) for ti in t])

    axs["time"].plot(t, cl, label=f"k={k}")
    axs["heave"].plot(coord_z[len(cl)//2:], cl[len(cl)//2:], label=f"k={k}")

uvlm_cl = []
uvlm_t = []
uvlm_z = []
with open("build/windows/x64/release/cl_data.txt", "r") as f:
    for line in f:
        t, z, cl = line.split()
        uvlm_t.append(float(t))
        uvlm_z.append(float(z))
        uvlm_cl.append(float(cl))

axs["time"].plot(uvlm_t, uvlm_cl, label="UVLM (k=0.5)", linestyle="--")
axs["time"].plot(uvlm_t, uvlm_z, label="UVLM z (k=0.5)", linestyle="--")

axs["heave"].plot(uvlm_z[len(uvlm_cl)//2:], uvlm_cl[len(uvlm_cl)//2:], label="UVLM (k=0.5)", linestyle="--")

axs["time"].set_xlabel('t')
axs["time"].set_ylabel('CL')
axs["time"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
axs["time"].legend()

axs["heave"].set_xlabel('h/c')
axs["heave"].set_ylabel('CL')
axs["heave"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
axs["heave"].legend()

plt.suptitle("Verification of UVLM with Theodorsen harmonic heave motion")
plt.show()

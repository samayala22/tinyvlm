import numpy as np
import scipy.integrate as spi
import scipy.special as scsp
from pathlib import Path
import plotting as plot
import plotly.graph_objects as go
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size': 20,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}"
})

EPS_sqrt_f = np.sqrt(1.19209e-07)

# Two point central difference
def derivative(f, x):
    return (f(x + EPS_sqrt_f) - f(x - EPS_sqrt_f)) / (2 * EPS_sqrt_f)

def derivative2(f, x):
    return (f(x + EPS_sqrt_f) - 2 * f(x) + f(x - EPS_sqrt_f)) / (EPS_sqrt_f ** 2)

def solve_ivp(x0: float, s0: float, sf: float, f: callable):
    return spi.solve_ivp(f, [s0, sf], [x0]).y[-1] # return only the result at t=sf

def theo_fun(k):
    H1 = scsp.hankel2(1, k)
    H0 = scsp.hankel2(0, k)
    C = H1 / (H1 + 1.0j * H0)
    return C

def uvlm_data(filename):
    uvlm_cl = []
    uvlm_t = []
    uvlm_z = []
    uvlm_alpha = []
    filepath = f"build/windows/x64/release/{filename}.txt"
    if Path(filepath).exists():
        with open(filepath, "r") as f:
            k = float(f.readline()) # get reduced frequency of the simulation
            for line in f:
                time_step, z, cl, alpha = line.split()
                uvlm_t.append(float(time_step))
                uvlm_z.append(float(z))
                uvlm_cl.append(float(cl))
                uvlm_alpha.append(float(alpha))
    else:
        print(f"File {filepath} not found")

    uvlm_alpha = np.array(uvlm_alpha)
    uvlm_cl = np.array(uvlm_cl)
    return k, uvlm_t, uvlm_z, uvlm_cl, uvlm_alpha

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
pitch_axis = -0.5 # quarter chord

# Theodorsen numerical simulation param
cycles = 3 # number of periods
nb_pts = 1000
cycle_idx = int((1 - 1 / cycles) * nb_pts - 1)

files = [
    "cl_data_CPU",
    # "cl_data_CUDA",
    # "cl_data_025",
    # "cl_data_050",
    # "cl_data_075",
]

for file in files:
    k, uvlm_t, uvlm_z, uvlm_cl, uvlm_alpha = uvlm_data(file)
    t_final = cycles * 2 * np.pi / (k * 2.0 * u_inf / c)
    omega = k * 2.0 * u_inf / c # frequency

    vec_t = np.linspace(0, t_final, nb_pts)

    n = len(uvlm_t) # number of time steps
    uvlm_cycle_idx = int((1 - 1 / cycles) * n - 2)

    # sudden acceleration
    # def pitch(t): return np.radians(5)
    # def heave(t): return 0

    # pure heaving
    # def pitch(t): return 0
    # def heave(t): return -0.1 * np.sin(omega * t)

    # pure pitching
    def pitch(t): return np.radians(1.0) *(np.sin(omega * t) + 2.0 * np.sin(3.0 * omega * t))
    def heave(t): return 0

    def w(s: float): 
        return u_inf * pitch(s) + derivative(heave, s) + b * (0.5 - pitch_axis) * derivative(pitch, s)

    def dx1ds(s: float, x1: float): return (u_inf / b) * (b1 * beta_1 * w(s) - beta_1 * x1)
    def dx2ds(s: float, x2: float): return (u_inf / b) * (b2 * beta_2 * w(s) - beta_2 * x2)

    x1_solution = spi.solve_ivp(dx1ds, [0, t_final], [0], t_eval=vec_t)
    x2_solution = spi.solve_ivp(dx2ds, [0, t_final], [0], t_eval=vec_t)
    
    def x1(s: float): return np.interp(s, x1_solution.t, x1_solution.y[0])
    def x2(s: float): return np.interp(s, x2_solution.t, x2_solution.y[0])

    def cl_theodorsen(t: float): # using Wagner functions and Kholodar formulation
        L_m = rho * b * b * np.pi * (u_inf * derivative(pitch, t) + derivative2(heave, t) - b * pitch_axis * derivative2(pitch, t))
        L_c = -2 * np.pi * rho * u_inf * b * (-(b0 + b1 + b2) * w(t) + x1(t) + x2(t))
        return (L_m + L_c) / (0.5 * rho * u_inf * u_inf * c)

    cl_theo = np.array([cl_theodorsen(ti) for ti in vec_t])
    h = np.array([-heave(ti) / c for ti in vec_t])
    angle = np.array([np.degrees(pitch(ti)) for ti in vec_t])

    fig_section = plot.fig_create_multi(1,1)
    fig_section.add_trace(
        go.Scatter(
            x = angle[cycle_idx:],
            y = cl_theo[cycle_idx:],
            mode='lines',
            line = {"dash": "solid", "width": 5, "simplify": False, "color": '#0066CC'},
            name="Theodorsen"
        )
    )
    fig_section.add_trace(
        go.Scatter(
            x = np.degrees(uvlm_alpha[uvlm_cycle_idx:]),
            y = uvlm_cl[uvlm_cycle_idx:],
            # mode='markers',
            # marker= {"size": 6, "color": "white", "line": {"width": 1, "color": "red"}},
            mode='lines',
            line = {"dash": "dash", "width": 5, "simplify": False, "color": "#FF8400"},
            name="UVLM"
        )
    )
    plot.format_subplot(fig_section, 1, 1, r"$\Large{\alpha} [\mathrm{deg}]$", r"$\Large{\mathrm{C_L}}$")
    plot.fig_save(fig_section, f"build/uvlm_{str(k).replace('.','')}")


    # plt.Figure(figsize=(4/3*500, 500))
    # plt.plot(angle[cycle_idx:], cl_theo[cycle_idx:], linewidth=4, label="Theodorsen")
    # plt.plot(np.degrees(uvlm_alpha[uvlm_cycle_idx:]), uvlm_cl[uvlm_cycle_idx:], linestyle=":", linewidth=4, label="UVLM")
    # plt.xlabel(r"$\alpha [\mathrm{deg}]$")
    # plt.ylabel(r"$\mathrm{CL}$")
    # plt.legend()
    # plt.savefig(f"build/uvlm_{str(k).replace('.','')}_mpl.pdf", bbox_inches='tight')

    analytical_cl = np.array([np.interp(ut, vec_t, cl_theo) for ut in uvlm_t[uvlm_cycle_idx:]])
    difference = uvlm_cl[uvlm_cycle_idx:] - analytical_cl
    error = np.sqrt(np.dot(difference, difference) / (n-uvlm_cycle_idx)) 
    print(f"Freq: {k}, Error: {100 * error / np.max(np.abs(analytical_cl)):.3f}%", )

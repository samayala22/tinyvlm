import dof3
import numpy as np
import scipy as sp
import plotting as plot
import plotly.graph_objects as go
from tqdm import tqdm

v = dof3.Vars()
v.a = -0.5 
v.b = 0.127 
v.c = 0.5 
v.I_alpha = 0.01347
v.I_beta = 0.0003264
v.k_h = 2818.8
v.k_alpha = 37.34
v.k_beta = 3.9
v.m = 1.5666
v.m_t = 3.39298
v.r_alpha = 0.7321
v.r_beta = 0.1140
v.S_alpha = 0.08587
v.S_beta = 0.00395
v.x_alpha = 0.4340
v.x_beta = 0.02
v.omega_h = 42.5352
v.omega_alpha = 52.6506
v.omega_beta = 109.3093
v.rho = 1.225
v.zeta_h = 0.0113
v.zeta_alpha = 0.01626
v.zeta_beta = 0.0115
v.sigma = v.omega_h / v.omega_alpha
v.mu = v.m / (np.pi * v.rho * v.b**2)

rms_samples = 50
t_final = 1000.0
dt = 0.1

U_flutter = 23.9
rms_param = np.linspace(0.2 * U_flutter, 0.9 * U_flutter, rms_samples)
rms_mat = np.zeros((8, rms_samples))
vec_t = np.arange(0, t_final, dt)
y0 = np.zeros(8, dtype=np.float64) # hd, ad, bd, h, a, b, x1, x2
y0[3] = 0.1 / v.b # h

for i, U in enumerate(tqdm(rms_param)):
    v.U = U
    v.V = v.U / (v.b * v.omega_alpha)
    system = dof3.AeroelasticSystem(v, True, dof3.alpha_freeplay)
    sol = sp.integrate.solve_ivp(system.coupled_system, (0, t_final), y0, t_eval=vec_t, method='RK45')

    idx_start = int(0.9 * len(sol.t))
    u_tr = sol.y[:, idx_start:]
    u_tr[3, :] *= v.b * 1e2 # h to cm
    u_tr[4, :] *= (180/np.pi) / 4.24 # alpha to deg and normalize by freeplay region
    u_tr[5, :] *= (180/np.pi) / 4.24 # beta to deg and normalize by freeplay region
    rms = np.sqrt(np.mean(u_tr**2, axis=1))
    rms_mat[:, i] = rms

dofs = ["h", "alpha", "beta"]
dofs_tex = [r"h", r"\alpha", r"\beta"]
conner = [np.loadtxt(f"scripts/conner_{d}.csv", delimiter=',', skiprows=1).T for d in dofs]

fig = plot.fig_create_multi(3, 1)
for i in range(3):
    fig.add_trace(
        go.Scatter(
            x=rms_param / U_flutter, 
            y=rms_mat[i+3, :],
            mode="lines",
            line={"color": "#636efa"},
            name = "Theodorsen",
            legendgroup = "Theodorsen",
            showlegend=True if i==0 else False,
        ),
        row=i + 1, 
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=conner[i][0, :], 
            y=conner[i][1, :],
            mode="markers",
            marker = {"size": 7, "color": "#ef553b"},
            name = "Conner Exp.",
            legendgroup = "Conner Exp.",
            showlegend=True if i==0 else False,
        ),
        row=i + 1, 
        col=1
    )
    plot.format_subplot(fig, i + 1, 1, r"$\Large{U / U_f}$", r"$\Large{\mathrm{RMS}(" + dofs_tex[i] + r")}$")
fig.update_xaxes(showticklabels=False, title_text="", row=1, col=1)
fig.update_xaxes(showticklabels=False, title_text="", row=2, col=1)
plot.fig_save(fig, f"build/3dof/3dof_freeplay_bifurcation_conner", html=False, height=1000)

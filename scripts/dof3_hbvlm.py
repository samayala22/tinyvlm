import sys

if sys.platform == "win32":
    sys.path.append(r".\\build\\windows\\x64\\release")
elif sys.platform == "linux":
    sys.path.append(r"./build/linux/x86_64/release")
else:
    exit()

import numpy as np
import scipy as sp
from dataclasses import dataclass
import pathlib 
import plotly.graph_objects as go

# local imports
import dof3
import helpers
import continuation as cont
import harmonic_balance as hb
import plotting as plot

from libhbvlm3 import HBVLM

BETA_NL_DAMPING = False
INITIAL_ONLY = False

@dataclass
class System:
    M      : callable
    C      : callable
    K      : callable
    dMdU   : callable
    dCdU   : callable
    dKdU   : callable
    fnlt   : callable        # time‐domain NL force
    fnlf   : callable        # frequency‐domain NL force
 
def nonlinear_damping(A, omega):
    """
    A: Amplitude
    omega: frequency in Hz
    """
    a3 = -1.1
    a2 = 2.0
    olog = np.log10(omega)
    return v.zeta_beta * np.max(a3 * olog**3 + a2 * olog**2 + 1, 0.2)

def create_motion_system() -> System:
    def fnlt(t, X, u, u_dot, omega, U):
        zero = np.zeros_like(t)
        if BETA_NL_DAMPING:
            freq = omega * v.omega_alpha # dimensional frequency
            A = np.sqrt(X[2, 1]**2 + X[2, 2]**2) # Amplitude of first harmonic
            beta_damping = - (v.omega_beta / v.omega_alpha) * v.r_beta**2 * nonlinear_damping(A, freq)
        else:
            beta_damping = zero
        beta_stiffness = - ((v.omega_beta / v.omega_alpha)**2 * v.r_beta**2) * torsional_func(u[2])
        return np.array([
            zero, # h
            zero, # alpha
            beta_stiffness # beta
        ])

    def fnlf(X, omega, U):
        assert omega > 0.0, "Omega must be positive"
        V = U / (v.b * v.omega_alpha)
        forces_t = np.zeros_like(X)
        hbvlm.run(omega, U, X, forces_t)
        forces_t[0, :] = - V*V * forces_t[0, :] / (np.pi * v.mu)
        forces_t[1, :] = 2.0*V*V * forces_t[1, :] / (np.pi * v.mu)
        forces_t[2, :] = (1-v.c)**2 * V * V * forces_t[2, :] / (2.0 * np.pi * v.mu)
        return forces_t
    
    def M(U):
        M_s = np.zeros((3,3))
        M_s[0, 0] = v.m_t / v.m
        M_s[0, 1] = v.x_alpha
        M_s[0, 2] = v.x_beta
        M_s[1, 0] = v.x_alpha
        M_s[1, 1] = v.r_alpha**2
        M_s[1, 2] = (v.c - v.a) * v.x_beta + v.r_beta**2
        M_s[2, 0] = v.x_beta
        M_s[2, 1] = (v.c - v.a) * v.x_beta + v.r_beta**2
        M_s[2, 2] = v.r_beta**2
        return M_s
    
    def C(U):
        D_s = np.zeros((3,3))
        D_s[0, 0] = v.sigma * v.zeta_h
        D_s[1, 1] = v.r_alpha**2 * v.zeta_alpha
        if not BETA_NL_DAMPING:
            D_s[2, 2] = (v.omega_beta / v.omega_alpha) * v.r_beta**2 * v.zeta_beta
        return 2 * D_s
    
    def K(U):
        K_s = np.zeros((3,3))
        K_s[0, 0] = v.sigma**2
        K_s[1, 1] = v.r_alpha**2
        # K_s[2, 2] = (v.omega_beta / v.omega_alpha)**2 * v.r_beta**2
        return K_s
    
    def dMdU(U): return np.zeros((3, 3))
    def dCdU(U): return np.zeros((3, 3))
    def dKdU(U): return np.zeros((3, 3))
    
    return System(M, C, K, dMdU, dCdU, dKdU, fnlt, fnlf)

def plot_hb_continuation2(theodorsen_metadata_list, hbvlm_metadata_list):
    filename = f"cont_3dof_{torsional_spring_names[torsional_spring]}_hbvlm_theodorsen_comparison"
    filedir = pathlib.Path(f"build/continuation/{filename}")
    filedir.mkdir(parents=True, exist_ok=True)
    dash = ["solid", "dot"]
    colors = ["#636efa", "#ef553b"]
    megalist = [theodorsen_metadata_list, hbvlm_metadata_list]
    labels = ["HB-Theodorsen", "HB-VLM"]
    dofs = 3

    for dof in range(dofs):
        fig = plot.fig_create_multi(1,1)
        for k in range(2):
            metadata_list = megalist[k]
            omega_idx = metadata_list[0].dims.n_u - metadata_list[0].X.shape[0]

            for i, md in enumerate(metadata_list):
                true_dof = dof if md.dims.n_d == 3 else dof + 3
                X_h = md.X[true_dof:omega_idx:md.dims.n_d, :]
                A = np.sqrt(X_h[1::2, :]**2 + X_h[2::2, :]**2)
                rms = np.sqrt(X_h[0, :]**2 + 0.5 * np.sum(A**2, axis=0))

                fig.add_trace(
                    go.Scatter(
                        x = md.X[-1, :],
                        y = rms,
                        name = labels[k],
                        mode = "lines",
                        line = {"dash": dash[k], "color": colors[k]},
                        showlegend = True if i == 0 else False
                    ),
                    row=1,
                    col=1
                )

        plot.format_subplot(fig, 1, 1, r"$\Large{U}$", r"$\Large{\mathrm{RMS}(x_{" + str(dof+1) + r"})}$")
        plot.fig_save(fig, filedir / f"{filename}_rms_{dof}", pdf=True)

    
    # Frequency plot
    fig = plot.fig_create_multi(1,1)
    for k in range(2):
        metadata_list = megalist[k]
        omega_idx = metadata_list[0].dims.n_u - metadata_list[0].X.shape[0]
        for i, md in enumerate(metadata_list):
            fig.add_trace(
                go.Scatter(
                    x = md.X[-1, :],
                    y = md.X[omega_idx, :],
                    name = labels[k],
                    mode = "lines",
                    line = {"dash": dash[k], "color": colors[k]},
                    showlegend = True if i == 0 else False
                ),
                row=1,
                col=1
            )
    plot.format_subplot(fig, 1, 1, r"$\Large{U}$", r"$\Large{\omega}$", ".1f")
    plot.fig_save(fig, filedir / f"{filename}_frequency", pdf=True)


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) == 4:
        torsional_spring = int(argv[1])
        param_start = float(argv[2])
        param_end = float(argv[3])
    else:
        torsional_spring = 0
        flutter_speed = 23.9
        # param_start = flutter_speed * 0.3
        # param_end = flutter_speed * 0.6
        param_start = 6.0
        param_end = 20.0
    
    torsional_spring_names = ["freeplay", "cubic", "linear"]

    if (torsional_spring == 0):
        torsional_func = dof3.alpha_freeplay
    elif (torsional_spring == 1):
        torsional_func = dof3.alpha_poly
    else:
        torsional_func = dof3.alpha_linear

    print(f"Param start: {param_start}, Param end: {param_end}")

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
    v.U = param_start
    v.V = v.U / (v.b * v.omega_alpha)
    v.mu = v.m / (np.pi * v.rho * v.b**2)

    # Independent params
    dims = hb.Dims(
        n_d=3,          # number of degrees of freedom
        n_h=5          # number of harmonics
    )

    hbvlm = HBVLM("cpu", ["mesh/3dof_wing_45x1.x", "mesh/3dof_flap_15x1.x"])
    # hbvlm = HBVLM("cpu", ["mesh/3dof_wing_9x5.x", "mesh/3dof_flap_3x5.x"])
    hbvlm.init(dims.n_h, 1.0/v.b)

    # Time integration
    t_final = 1000.0
    dt = 0.1 
    vec_t = np.arange(0, t_final, dt)
    y0 = np.zeros(8, dtype=np.float64) # hd, ad, bd, h, a, b, x1, x2
    y0[3] = 0.01 / v.b # h
    system = dof3.AeroelasticSystem(v, True, torsional_func)
    sol = sp.integrate.solve_ivp(system.coupled_system, (0, t_final), y0, t_eval=vec_t, method='RK45')

    idx_start = int(0.9 * len(sol.t))
    t_tr = sol.t[idx_start:]
    u_tr = sol.y[3:6, idx_start:]   # shape = (n_dofs, N_tr)
    u_coeffs, omega0 = hb.truncated_series_approximation(dt, u_tr, dims)
    X0 = np.zeros(dims.n_d * dims.n_c + 2)
    X0[:-2] = u_coeffs.T.reshape(-1)
    X0[-2] = omega0
    X0[-1] = param_start

    # temporary
    if torsional_spring == 1:
        md = cont.load_metadata("build/cont_3dof_hbvlm_cubic_st_9_end_20_it_166.pkl")
        idx = np.argmax(md.X[-1, :])
        X0 = md.X[:, idx]
        X0[-1] = param_start

    if torsional_spring == 0:
        md = cont.load_metadata("build/cont_3dof_hbvlm_freeplay_st_6_end_20_it_111.pkl")
        X0 = md.X[:, -1]

    metadata = cont.Metadata()
    metadata.name = f"3dof_hbvlm_{torsional_spring_names[torsional_spring]}"
    metadata.param_start = param_start
    metadata.param_end = param_end
    metadata.max_steps = 1 if INITIAL_ONLY else 5000
    metadata.scaling = True
    metadata.step_adapt = True
    metadata.ds = 0.02
    metadata.dims = dims
    
    motion = create_motion_system()
    if not helpers.getenv("SKIP"):
        metadata = cont.continuation(X0, motion, metadata)
    
    if helpers.getenv("DEBUG"):
        # metadata = cont.load_metadata("build/cont_3dof_hbvlm_cubic_st_16_end_17_it_39.pkl")
        it = 0
        omega_idx = -2
        X = metadata.X[:, it]
        hb_sol_t, hb_sol = hb.to_timedomain(vec_t, dims.n_d, X[:-2], X[-2], dims.n_h)
        
        R_nlft = motion.fnlf(X[:omega_idx].reshape(dims.n_c, dims.n_d).T, X[-2], X[-1]) # dot x n_c
        # R_nlft = motion.fnlf(X0[:omega_idx].reshape(dims.n_c, dims.n_d).T, X0[-2], X0[-1]) # dot x n_c
        
        R_nlf_fft = np.fft.rfft(R_nlft, dims.n_c, axis=1, norm='backward')
        R_nlf = hb.X_to_real(R_nlf_fft / dims.n_c, 0.0).T.reshape(-1)
        hbf_t, hbf = hb.to_timedomain(vec_t, dims.n_d, R_nlf, X[-2], dims.n_h)
        
        aero_forces = system.aero_forces(sol.y)
        fig = plot.create_dofs_figure(["Heave", "Pitch", "Control"])
        dof3.plot_solution(fig, aero_forces, sol, v)

        plot.add_data_and_psd(fig, hb_sol_t, hb_sol[0, :], "HB-VLM", 1, 1, 3)
        plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol[1, :]), "HB-VLM", 3, 1, 3)
        plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol[2, :]), "HB-VLM", 5, 1, 3)

        plot.add_data_and_psd(fig, hb_sol_t, hb_sol[3, :], "HB-VLM", 1, 2, 3)
        plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol[4, :]), "HB-VLM", 3, 2, 3)
        plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol[5, :]), "HB-VLM", 5, 2, 3)

        plot.add_data_and_psd(fig, hbf_t, hbf[0, :], "HB-VLM", 1, 3, 3)
        plot.add_data_and_psd(fig, hbf_t, hbf[1, :], "HB-VLM", 3, 3, 3)
        plot.add_data_and_psd(fig, hbf_t, hbf[2, :], "HB-VLM", 5, 3, 3)
        
        dof3.format_plot(fig)
        plot.fig_save(fig, f"build/3dof/hbvlm_{it}", html=True, pdf=False)

    if not helpers.getenv("POST"):
        exit(0)
    if torsional_spring == 1:
        theodorsen_metadata_files = [
            "build/cont_3dof_cubic_st_6_end_20_it_285.pkl",
            "build/cont_3dof_cubic_st_6_end_1_it_326.pkl",
            "build/cont_3dof_cubic_st_12_end_20_it_212.pkl",
            "build/cont_3dof_cubic_st_12_end_10_it_161.pkl",
            "build/cont_3dof_cubic_st_11_end_1_it_405.pkl" # went back and forth
        ]
        hbvlm_metadata_files = [
            "build/cont_3dof_hbvlm_cubic_st_12_end_20_it_140.pkl",
            "build/cont_3dof_hbvlm_cubic_st_12_end_10_it_168.pkl",
            "build/cont_3dof_hbvlm_cubic_st_6_end_10_it_66.pkl",
            "build/cont_3dof_hbvlm_cubic_st_6_end_1_it_75",
            "build/cont_3dof_hbvlm_cubic_st_9_end_20_it_166.pkl",
            "build/cont_3dof_hbvlm_cubic_st_14_end_20_it_20.pkl",
            "build/cont_3dof_hbvlm_cubic_st_14_end_12_it_47.pkl"
        ]
    elif torsional_spring == 0:
        theodorsen_metadata_files = [
            "build/cont_3dof_freeplay_st_6_end_20_it_284.pkl",
            "build/cont_3dof_freeplay_st_6_end_1_it_808.pkl",
            "build/cont_3dof_freeplay_st_15_end_20_it_157.pkl",
            "build/cont_3dof_freeplay_st_15_end_1_it_495.pkl",
            "build/cont_3dof_freeplay_st_11_end_20_it_752.pkl",
            "build/cont_3dof_freeplay_st_11_end_9_it_85.pkl"
        ]
        hbvlm_metadata_files = [
            "build/cont_3dof_hbvlm_freeplay_st_6_end_20_it_111.pkl",
            "build/cont_3dof_hbvlm_freeplay_st_6_end_1_it_215.pkl",
            "build/cont_3dof_hbvlm_freeplay_st_15_end_20_it_45.pkl",
            "build/cont_3dof_hbvlm_freeplay_st_15_end_1_it_93.pkl"
        ]
    
    plot_hb_continuation2(
        [cont.load_metadata(f) for f in theodorsen_metadata_files],
        [cont.load_metadata(f) for f in hbvlm_metadata_files]
    )

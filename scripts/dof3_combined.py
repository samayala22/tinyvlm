import sys
import numpy as np
import scipy as sp

import dof3
import continuation as cont

import pathlib
import plotly.graph_objects as go
import plotting as plot

def plot_hb_continuation2(torsional_spring, theodorsen5_metadata_list, theodorsen3_metadata_list, theodorsen_metadata_list, timeseries=None):
    torsional_spring_names = ["freeplay", "cubic", "linear"]
    filename = f"cont_3dof_{torsional_spring_names[torsional_spring]}_theodorsenH_comparison"
    filedir = pathlib.Path(f"build/continuation/{filename}")
    filedir.mkdir(parents=True, exist_ok=True)
    dash = ["solid", "dot", "dash"]
    colors = ["#636efa", "#ef553b", "#00cc96"]
    megalist = [theodorsen5_metadata_list, theodorsen3_metadata_list, theodorsen_metadata_list]
    labels = ["HB (H=5)", "HB (H=3)", "HB (H=1)"]
    dofs = 3
    dof_name = [r"h", r"\alpha", r"\beta"]

    for dof in range(dofs):
        fig = plot.fig_create_multi(1,1)
        for k in range(len(megalist)):
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

        if timeseries is not None:
            ts_param, ts_rms = timeseries
            fig.add_trace(
                go.Scatter(
                    x = ts_param,
                    y = ts_rms[dof+3, :],
                    name = "TI",
                    mode = "markers",
                    marker = {"size": 10, "symbol": "star-diamond", "color": "#FFD700"},
                    showlegend=True
                ),
                row=1,
                col=1
            )

        plot.format_subplot(fig, 1, 1, r"$\Large{U}$", r"$\Large{\mathrm{RMS}(" + dof_name[dof] + r")}$")
        plot.fig_save(fig, filedir / f"{filename}_rms_{dof}", pdf=True)
    
    # Frequency plot
    fig = plot.fig_create_multi(1,1)
    for k in range(len(megalist)):
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

def plot_combined(torsional_spring:int):
    torsional_func = [dof3.alpha_freeplay, dof3.alpha_poly, dof3.alpha_linear][torsional_spring]

    rms_samples = 20
    rms_param = np.linspace(5.0, 20.0, rms_samples)
    rms_mat = np.zeros((8, rms_samples))
    v = dof3.Vars()
    for i, U in enumerate(rms_param):
        v = dof3.update_vars(v, U)
        t_final = 1000.0
        dt = 0.1 
        vec_t = np.arange(0, t_final, dt)
        y0 = np.zeros(8, dtype=np.float64) # hd, ad, bd, h, a, b, x1, x2
        y0[3] = 0.01 / v.b # h
        system = dof3.AeroelasticSystem(v, True, torsional_func)
        sol = sp.integrate.solve_ivp(system.coupled_system, (0, t_final), y0, t_eval=vec_t, method='RK45')

        idx_start = int(0.9 * len(sol.t))
        u_tr = sol.y[:, idx_start:]
        rms = np.sqrt(np.mean(u_tr**2, axis=1))
        rms_mat[:, i] = rms

    if torsional_spring == 1:
        # H=1 (fails to capture secondary branch)
        metadata_files = [
            "build/cont_3dof_cubic_st_6_end_20_it_135.pkl",
            "build/cont_3dof_cubic_st_6_end_1_it_105.pkl",
            "build/cont_3dof_cubic_st_12_end_20_it_120.pkl",
            "build/cont_3dof_cubic_st_12_end_10_it_125.pkl",
        ]

        # H=3
        metadata3_files = [
            "build/cont_3dof_cubic_st_6_end_20_it_175_h3.pkl",
            "build/cont_3dof_cubic_st_6_end_1_it_135_h3.pkl",
            "build/cont_3dof_cubic_st_12_end_20_it_163_h3.pkl",
            "build/cont_3dof_cubic_st_12_end_10_it_167_h3.pkl",
            "build/cont_3dof_cubic_st_11_end_1_it_155_h3.pkl"
        ]

        # H=5
        metadata5_files = [
            "build/cont_3dof_cubic_st_6_end_20_it_285.pkl",
            "build/cont_3dof_cubic_st_6_end_1_it_326.pkl",
            "build/cont_3dof_cubic_st_12_end_20_it_212.pkl",
            "build/cont_3dof_cubic_st_12_end_10_it_161.pkl",
            "build/cont_3dof_cubic_st_11_end_1_it_405.pkl" # went back and forth
        ]
    elif torsional_spring == 0:
        # H=1
        metadata_files = [
            "build/cont_3dof_freeplay_st_6_end_20_it_122.pkl",
            "build/cont_3dof_freeplay_st_6_end_1_it_157.pkl",
            "build/cont_3dof_freeplay_st_15_end_20_it_72.pkl",
            "build/cont_3dof_freeplay_st_15_end_1_it_103.pkl"
        ]

        # H=3
        metadata3_files = [
            "build/cont_3dof_freeplay_st_6_end_20_it_162.pkl",
            "build/cont_3dof_freeplay_st_6_end_1_it_217_h3.pkl",
            "build/cont_3dof_freeplay_st_15_end_20_it_104.pkl",
            "build/cont_3dof_freeplay_st_15_end_1_it_142_h3.pkl",
            "build/cont_3dof_freeplay_st_11_end_20_it_475_h3.pkl"
        ]

        # H=5
        metadata5_files = [
            "build/cont_3dof_freeplay_st_6_end_20_it_284.pkl",
            "build/cont_3dof_freeplay_st_6_end_1_it_808.pkl",
            "build/cont_3dof_freeplay_st_15_end_20_it_157.pkl",
            "build/cont_3dof_freeplay_st_15_end_1_it_495.pkl",
            "build/cont_3dof_freeplay_st_11_end_20_it_752.pkl",
            "build/cont_3dof_freeplay_st_11_end_9_it_85.pkl"
        ]
    
    metadatas = []
    import pickle
    for filename in metadata_files:
        with open(filename, 'rb') as f:
            metadatas.append(pickle.load(f))
        print(f"Loaded: {filename}")
    # cont.plot_hb_continuation(metadatas, timeseries=(rms_param, rms_mat))

    metadatas3 = []
    for filename in metadata3_files:
        with open(filename, 'rb') as f:
            metadatas3.append(pickle.load(f))
        print(f"Loaded: {filename}")
    # cont.plot_hb_continuation(metadatas3, timeseries=(rms_param, rms_mat))

    metadatas5 = []
    for filename in metadata5_files:
        with open(filename, 'rb') as f:
            metadatas5.append(pickle.load(f))
        print(f"Loaded: {filename}")
    # cont.plot_hb_continuation(metadatas5, timeseries=(rms_param, rms_mat))

    plot_hb_continuation2(torsional_spring, metadatas5, metadatas3, metadatas, timeseries=(rms_param, rms_mat))

if __name__ == "__main__":
    argv = sys.argv
    assert len(argv) == 2, "Usage: python dof3_combined.py <torsional_spring_type>"
    torsional_spring = int(argv[1])
    plot_combined(torsional_spring)
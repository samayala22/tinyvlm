import numpy as np
import plotly.graph_objects as go
import pickle
# local imports
import helpers
import continuation as cont
import harmonic_balance as hb
import plotting as plot
import integrators as integrator

def create_motion_system():
    def fnlt(t, X, u, v, omega, U):
        return np.array([- U * u[0]**2 * v[0], - U * u[1]**2 * v[1]])

    def fnlf(X, omega, U):
        forces_t = np.zeros_like(X)
        return forces_t
    
    def M(U):
        theta = 1.0
        return np.array([[theta, 0.0], [0.0, theta]])
    
    def C(U):
        return np.array([[-U, 0.0], [0.0, -U]])
    
    def K(U):
        kappa = 1.0
        return np.array([[1 + kappa, - kappa], [- kappa, 1 + kappa]])
    
    def dMdU(U): return np.zeros((2, 2))
    def dCdU(U): return np.zeros((2, 2))
    def dKdU(U): return np.zeros((2, 2))
    
    return cont.System(M, C, K, dMdU, dCdU, dKdU, fnlt, fnlf)

def integrate_motion_system(motion, t_final, dt, mu):
    u0 = np.zeros(2)
    v0 = np.zeros(2)
    # u0[0] = 1.0
    u0[0] = 3.0
    u0[1] = 3.0
    nl_func = lambda t, u, v: motion.fnlt(t, None, u, v, None, mu)
    t, u, v, a = integrator.nonlinear_newmark_solve(motion.M(mu), motion.C(mu), motion.K(mu), u0, v0, nl_func, t_final, dt)
    return t, u, v

def plot_continuation(metadata, rms_mu, rms_mat):
    dofs = metadata.dims.n_d
    omega_idx = metadata.dims.n_u - metadata.X.shape[0]
    for dof in range(dofs):
        X_h = metadata.X[dof:omega_idx:dofs, :]
        A = np.sqrt(X_h[1::2, :]**2 + X_h[2::2, :]**2)
        rms = np.sqrt(X_h[0, :]**2 + 0.5 * np.sum(A**2, axis=0))

        fig = plot.fig_create_multi(1,1)
        fig.add_trace(
            go.Scatter(
                x = metadata.X[-1, :],
                y = rms,
                name = "HB",
                mode = "lines",
                line = {"dash": "solid"},
                showlegend=True
            )
        )
        fig.add_trace(
            go.Scatter(
                x = rms_mu,
                y = rms_mat[dof, :],
                name = "TI",
                mode = "markers",
                showlegend=True
            )
        )
        plot.format_subplot(fig, 1, 1, r"$\Large{\mu}$", r"$\Large{RMS(x_{" + str(dof+1) + r"})}$")
        plot.fig_save(fig, f"build/vanderpol/vdp_rms_{dof}")

def plot_continuation_omega(metadata):
    fig = plot.fig_create_multi(1,1)
    fig.add_trace(
        go.Scatter(
            x = metadata.X[-1, :],
            y = metadata.X[-2, :],
            name = "HB",
            mode = "lines",
            line = {"dash": "solid"},
            showlegend=True
        )
    )
    plot.format_subplot(fig, 1, 1, r"$\Large{\mu}$", r"$\Large{\omega}$")
    plot.fig_save(fig, f"build/vanderpol/vdp_omega")

def plot_poincare_section(metadata: cont.Metadata, nb_points=5):
    step = int(metadata.X.shape[1] / nb_points)
    for i in range(nb_points):
        j = i * step
        mu = metadata.X[-1, j]
        print(f"i: {i}, mu: {mu}")
        t, u, v = integrate_motion_system(motion, 2000.0, 0.01, mu)
        idx_start = int(0.9 * len(t))
        u_tr = u[:, idx_start:]
        v_tr = v[:, idx_start:]

        T = 2 * np.pi / metadata.X[-2, j]
        _, hb_sol0 = hb.to_timedomain(np.linspace(0, T, 5000), dims.n_d, metadata.X[:omega_idx, j], metadata.X[omega_idx, j], dims.n_h)
        hb_u = hb_sol0[0:2, :]
        hb_v = hb_sol0[2:4, :]

        fig = plot.fig_create_multi(1,1)
        fig.add_trace(
            go.Scatter(
                x = u_tr[0, :],
                y = v_tr[0, :],
                name = "TI",
                mode = "lines",
                line = {"dash": "solid", "width": 8, "simplify": True}
            )
        )
        fig.add_trace(
            go.Scatter(
                x = hb_u[0, :],
                y = hb_v[0, :],
                name = "HB",
                mode = "lines",
                line = {"dash": "solid", "width": 4, "simplify": True}
            )
        )
        plot.format_subplot(fig, 1, 1, r"$\Large{x}$", r"$\Large{\dot{x}}$")
        plot.fig_save(fig, f"build/vanderpol/vdp_poincare_{i}")

if __name__ == "__main__":
    INITIAL_ONLY = 0
    # Continuation param is mu, nonlinear damping factor
    param_beg = 0.1
    param_end = 5.0
    
    # Independent params
    dims = hb.Dims(
        n_d=2,          # number of degrees of freedom
        n_h=30          # number of harmonics
    )
    omega_idx = -2

    # Time integration
    motion = create_motion_system()
    t_final = 2000.0
    dt = 0.01
    vec_t = np.arange(0, t_final, dt)

    t, u, v = integrate_motion_system(motion, t_final, dt, param_beg)

    idx_start = int(0.75 * len(t))
    t_tr = t[idx_start:]
    u_tr = u[:, idx_start:]
    u_coeffs, omega0 = hb.truncated_series_approximation(dt, u_tr, dims)
    X0 = np.zeros(dims.n_d * dims.n_c + 2)
    X0[:omega_idx] = u_coeffs.T.reshape(-1)
    X0[omega_idx] = omega0
    X0[-1] = param_beg

    metadata = cont.Metadata()
    metadata.name = f"Van der Pol Oscillator"
    metadata.param_start = param_beg
    metadata.param_end = param_end
    metadata.max_steps = 1 if INITIAL_ONLY else 10000
    metadata.scaling = True
    metadata.step_adapt = True
    metadata.ds = 0.01
    metadata.dims = dims
    
    metadata = cont.continuation(X0, motion, metadata)
    
    # X_mat = metadata.X
    # if X_mat.shape[1] == 1:
    #     hb_sol_t, hb_sol0 = hb.to_timedomain(vec_t, dims.n_d, X_mat[:omega_idx, 0], X_mat[omega_idx, 0], dims.n_h)
    #     _, hb_sol_approx = hb.to_timedomain(vec_t, dims.n_d, X0[:omega_idx], X0[omega_idx], dims.n_h)

    #     fig = plot.create_dofs_figure(["dof1", "dof2"])
    #     plot.add_data_and_psd(fig, t, u[0, :], "Time Integration", 1, 1)
    #     plot.add_data_and_psd(fig, t, u[1, :], "Time Integration", 3, 1)
    #     plot.add_data_and_psd(fig, t, v[0, :], "Time Integration", 1, 2)
    #     plot.add_data_and_psd(fig, t, v[1, :], "Time Integration", 3, 2)

    #     plot.add_data_and_psd(fig, t, hb_sol_approx[0, :], "Approx", 1, 1, 1)
    #     plot.add_data_and_psd(fig, t, hb_sol_approx[1, :], "Approx", 3, 1, 1)
    #     plot.add_data_and_psd(fig, t, hb_sol_approx[2, :], "Approx", 1, 2, 1)
    #     plot.add_data_and_psd(fig, t, hb_sol_approx[3, :], "Approx", 3, 2, 1)

    #     plot.add_data_and_psd(fig, t, hb_sol0[0, :], "HB", 1, 1, 2)
    #     plot.add_data_and_psd(fig, t, hb_sol0[1, :], "HB", 3, 1, 2)
    #     plot.add_data_and_psd(fig, t, hb_sol0[2, :], "HB", 1, 2, 2)
    #     plot.add_data_and_psd(fig, t, hb_sol0[3, :], "HB", 3, 2, 2)
    #     plot.fig_save(fig, f"build/vanderpol/vdp0")
        
    # else:
    #     cont.plot_hb_continuation(metadata)

    # RAW
    # with open("build/continuation_c4cffd4d.pkl", 'rb') as f:
    #     metadata = pickle.load(f)

    # With lanczos m=0.5
    # with open("build/continuation_7d34371a.pkl", 'rb') as f:
    #     metadata = pickle.load(f)

    # rms_samples = 20
    # rms_mu = np.linspace(param_beg, param_end, rms_samples)
    # rms_mat = np.zeros((dims.n_d, rms_samples))
    # for i, mu in enumerate(rms_mu):
    #     t, u, v = integrate_motion_system(motion, t_final, dt, mu)
    #     idx_start = int(0.75 * len(t))
    #     u_tr = u[:, idx_start:]
    #     rms = np.sqrt(np.mean(u_tr**2, axis=1))
    #     rms_mat[:, i] = rms

    #     fig = plot.create_dofs_figure(["dof1", "dof2"])
    #     plot.add_data_and_psd(fig, t, u[0, :], "Time Integration", 1, 1)
    #     plot.add_data_and_psd(fig, t, u[1, :], "Time Integration", 3, 1)
    #     plot.add_data_and_psd(fig, t, v[0, :], "Time Integration", 1, 2)
    #     plot.add_data_and_psd(fig, t, v[1, :], "Time Integration", 3, 2)
    #     plot.fig_save(fig, f"build/vanderpol/vdp_{i}", pdf=False)

    # plot_continuation(metadata, rms_mu, rms_mat)
    # plot_continuation_omega(metadata)
    plot_poincare_section(metadata, 5)



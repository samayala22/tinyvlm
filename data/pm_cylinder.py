import plotly.graph_objects as go
import numpy as np

if __name__ == "__main__":
    pm_theta = []
    pm_cp = []
    with open("build/windows/x64/debug/pm_cylinder.txt", "r") as f:
        for line in f:
            theta, cp = map(float, line.split())
            pm_theta.append(theta)
            pm_cp.append(cp)
    pm_theta = np.array(pm_theta)
    pm_cp = np.array(pm_cp)
    sort_idx = np.argsort(pm_theta)
    pm_theta_sorted = pm_theta[sort_idx]
    pm_cp_sorted = pm_cp[sort_idx]

    analytical_theta = np.linspace(-np.pi, np.pi, 100)
    analytical_cp = 1 - 4 * np.sin(analytical_theta)**2

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=pm_theta_sorted, y=pm_cp_sorted, mode='lines+markers', line=dict(color='red')))
    fig.add_trace(go.Scattergl(x=analytical_theta, y=analytical_cp, mode='lines', line=dict(color='blue')))
    fig.update_layout(title='Pressure Coefficient')
    fig.show()
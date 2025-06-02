import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def plot_uvlm(fig):
    uvlm_t = []
    uvlm_gamma = []
    uvlm_cl = []
    uvlm_cm = []
    uvlm_dgamma = []
    file_path = Path("build/windows/x64/release/uvlm_data_CPU.txt")
    if not file_path.exists():
        print("UVLM data not found")
        return
    with open(file_path, "r") as f:
        for line in f:
            t, gamma, cl, cm, dgamma = map(float, line.split())
            uvlm_t.append(t)
            uvlm_gamma.append(gamma)
            uvlm_cl.append(cl)
            uvlm_cm.append(cm)
            uvlm_dgamma.append(dgamma)

    common_settings = dict(
        name="UVLM",
        mode='markers',
        marker=dict(size=4),
        showlegend=True
    )

    traces_data = [
        (uvlm_t, uvlm_gamma, 1, 1),
        (uvlm_t, uvlm_dgamma, 4, 1),
        (uvlm_t, uvlm_cl, 2, 1),
        (uvlm_t, uvlm_cm, 3, 1)
    ]

    for x, y, row, col in traces_data:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                **common_settings
            ),
            row=row,
            col=col
        )

def plot_hbvlm(fig):
    hbvlm_t = []
    hbvlm_gamma = []
    hbvlm_cl = []
    hbvlm_cm = []
    hbvlm_dgamma = []
    filepath = Path("build/windows/x64/release/hbvlm_data_CPU.txt")
    if not filepath.exists():
        print("HBVLM data not found")
        return
    with open(filepath, "r") as f:
        for line in f:
            t, gamma, cl, cm, dgamma = map(float, line.split())
            hbvlm_t.append(t)
            hbvlm_gamma.append(gamma)
            hbvlm_cl.append(cl)
            hbvlm_cm.append(cm)
            hbvlm_dgamma.append(dgamma)

    common_settings = dict(
        name="HBVLM",
        mode='markers',
        marker=dict(size=4),
        showlegend=True
    )

    traces_data = [
        (hbvlm_t, hbvlm_gamma, 1, 1),
        (hbvlm_t, hbvlm_dgamma, 4, 1),
        (hbvlm_t, hbvlm_cl, 2, 1),
        (hbvlm_t, hbvlm_cm, 3, 1)
    ]

    for x, y, row, col in traces_data:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                **common_settings
            ),
            row=row,
            col=col
        )

if __name__ == "__main__":
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "gamma", "cl", "cm", "dgamma/dt"
        ),
    )

    plot_hbvlm(fig)
    plot_uvlm(fig)

    fig.update_layout(
        title="UVLM vs HBVLM",
        title_x=0.5,
        autosize=True,
        showlegend=True,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.0
        )
    )

    fig.write_html("build/hbvlm_verif.html", include_mathjax='cdn')
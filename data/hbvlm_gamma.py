import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    uvlm_t = []
    uvlm_gamma = []
    with open("build/windows/x64/release/uvlm_gamma_CPU.txt", "r") as f:
        for line in f:
            t, gamma = map(float, line.split())
            uvlm_t.append(t)
            uvlm_gamma.append(gamma)

    hbvlm_t = []
    hbvlm_gamma = []
    with open("build/windows/x64/release/hbvlm_gamma_CPU.txt", "r") as f:
        for line in f:
            t, gamma = map(float, line.split())
            hbvlm_t.append(t)
            hbvlm_gamma.append(gamma)

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(
            "Gammas"
        ),
    )

    fig.add_trace(
        go.Scattergl(
            x=uvlm_t, 
            y=uvlm_gamma,
            name="UVLM",
            mode='markers',
            marker=dict(size=4),
            showlegend=True
        ),
        row=1, 
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=hbvlm_t, 
            y=hbvlm_gamma,
            name="HBVLM",
            mode='markers',
            marker=dict(size=4),
            showlegend=True
        ),
        row=1, 
        col=1
    )

    fig.update_layout(
        title="Gamma",
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

    fig.write_html("build/hbvlm_gamma.html", include_mathjax='cdn')
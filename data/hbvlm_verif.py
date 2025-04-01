import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    uvlm_t = []
    uvlm_gamma = []
    uvlm_cl = []
    uvlm_cm = []
    with open("build/windows/x64/release/uvlm_data_CPU.txt", "r") as f:
        for line in f:
            t, gamma, cl, cm = map(float, line.split())
            uvlm_t.append(t)
            uvlm_gamma.append(gamma)
            uvlm_cl.append(cl)
            uvlm_cm.append(cm)

    hbvlm_t = []
    hbvlm_gamma = []
    hbvlm_cl = []
    hbvlm_cm = []
    with open("build/windows/x64/release/hbvlm_data_CPU.txt", "r") as f:
        for line in f:
            t, gamma, cl, cm = map(float, line.split())
            hbvlm_t.append(t)
            hbvlm_gamma.append(gamma)
            hbvlm_cl.append(cl)
            hbvlm_cm.append(cm)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "gamma", "cl", "cm"
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

    fig.add_trace(
        go.Scattergl(
            x=hbvlm_t, 
            y=hbvlm_cl,
            name="HBVLM",
            mode='markers',
            marker=dict(size=4),
            showlegend=True
        ),
        row=2, 
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=uvlm_t, 
            y=uvlm_cl,
            name="UVLM",
            mode='markers',
            marker=dict(size=4),
            showlegend=True
        ),
        row=2, 
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=hbvlm_t,
            y=hbvlm_cm,
            name="HBVLM",
            mode='markers',
            marker=dict(size=4),
            showlegend=True
        ),
        row=3, 
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=uvlm_t,
            y=uvlm_cm,
            name="UVLM",
            mode='markers',
            marker=dict(size=4),
            showlegend=True
        ),
        row=3, 
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

    fig.write_html("build/hbvlm_verif.html", include_mathjax='cdn')
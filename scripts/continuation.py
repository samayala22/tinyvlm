import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

style_dict = {
    "layout.font.family": "Times New Roman",
    "layout.font.size": 16,
    "layout.template": "plotly_white",
}

def plot_hb_continuation(title, H, X_mat):
    assert len(X_mat.shape) == 2
    dofs = (X_mat.shape[0] - 2) / (2*H+1)
    assert dofs.is_integer()
    dofs = int(dofs)
    print(f"dofs: {dofs}")
    
    fig = make_subplots(
        rows=dofs, cols=1,
        subplot_titles=tuple(f"DOF {i+1}" for i in range(dofs)),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    fig.update_layout(title=title)
    fig.update(**style_dict)

    X_mat_h = X_mat[:-2, :]
    for dof in range(dofs):
        X_h = X_mat_h[dof::dofs, :]
        for h in range(1, H+1):
            fig.add_trace(
                go.Scattergl(
                    x = X_mat[-1, :],
                    y = np.sqrt(X_h[2*h-1, :]**2 + X_h[2*h, :]**2),
                    name = f"Harmonic {h}",
                    mode = "lines+markers"
                ),
                row=dof+1,
                col=1
            )
    fig.update_xaxes(title_text=r"$\bar{U}$", showgrid=True)
    fig.update_yaxes(title_text=r"$||H_{j}||^{2}$", showgrid=True)
    fig.write_html("build/continuation.html", include_mathjax='cdn')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("H", type=int)
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    plot_hb_continuation("Continuation", args.H, np.load(args.filename))
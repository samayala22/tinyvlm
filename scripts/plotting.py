from plotly.subplots import make_subplots
import plotly.graph_objects as go
import kaleido
import numpy as np
import scipy as sp
from pathlib import Path
import helpers

COLORS = ['royalblue', 'orange', 'green', 'red', 'purple']

def fig_dims(fig):
    rows_range, cols_range = fig._get_subplot_rows_columns()
    return list(rows_range)[-1]+1, list(cols_range)[-1]+1

@helpers.measure
def fig_create_multi(rows, cols, subplot_titles: tuple[str] = "", fig_title: str = ""):
    fig = make_subplots(
        rows=rows, cols=cols,
        # subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        # shared_xaxes=True,
        vertical_spacing=0.04,
    )
    
    fig.update_layout(
        # title=fig_title,
        title_x=0.5,
        # showlegend=True,
        font_family="Latin Modern Roman",
        font_size=25,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01
        )
    )
    return fig

def create_dofs_figure(dof_names: list[str], title: str = ""):
    return fig_create_multi(len(dof_names)*2, 3, tuple(f"{dof_name} {var} {psd}" for dof_name in dof_names for psd in ["", "PSD"] for var in ["Position", "Velocity", "Force"]), title)

@helpers.measure
def fig_save(fig, filename, html=False, pdf=True, height=500):
    print(f"Saving {filename} ...")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    ratio = 4/3
    width = int(ratio * height)
    fig.update_layout(autosize=True)
    if html:
        fig.write_html(f"{filename}.html", include_mathjax='cdn')
    if not pdf:
        return
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(
            l=10,   # Left margin
            r=10,   # Right margin  
            t=10,   # Top margin
            b=10,   # Bottom margin
            pad=0   # Padding between plot and margins
    ))
    kaleido.write_fig_sync(fig, path=f"{filename}.pdf")

plotly_axes = {
    "showgrid": True,
    "gridwidth": 1,
    "griddash": "dot",
    "gridcolor": 'rgba(128, 128, 128, 0.2)',
    "showline": True,
    "linecolor": "black",
    "linewidth": 1,
    "mirror": True,
    "ticks": "inside",
    "ticklen": 8,
    "tickwidth": 1,
    "tickcolor": "black"
}

def format_subplot(fig, row, col, xlabel, ylabel):
    """Format a specific subplot with labels and grid"""
    fig.update_xaxes(
        title_text=xlabel,
        row=row,
        col=col,
        matches= f"x{1 if row % 2 == 1 else 4}",
        **plotly_axes
    )
    fig.update_yaxes(
        title_text=ylabel,
        row=row,
        col=col,
        tickformat=".1e",
        **plotly_axes
    )

def compute_psd(t, data):
    """Compute PSD with consistent parameters"""
    sampling_rate = 1 / np.mean(np.diff(t))

    frequencies, psd = sp.signal.welch(
        data[int(0.75*len(data)):],
        nperseg=int(0.25 * len(data)),
        nfft=len(data),
        fs=sampling_rate,
        window='boxcar',
        scaling='spectrum'
    )

    mask = frequencies < 1.0
    # psd_db = 10 * np.log10(psd)
    return frequencies[mask], psd[mask]

def find_peak_idx(data):
    """Find indices of peaks and valleys in data"""
    peaks_idx0, _ = sp.signal.find_peaks(data)  # peaks
    peaks_idx1, _ = sp.signal.find_peaks(-data)  # valleys
    peaks_idx = np.concatenate((peaks_idx0, peaks_idx1))
    return peaks_idx

def add_data(fig, time, data, name, row, col, data_id=0, mode='lines', marker_size=4):
    assert data_id < len(COLORS)
    line = {"color": COLORS[data_id]}

    fig.add_trace(
        go.Scatter(
            x=time, 
            y=data, 
            name=name,
            legendgroup=name,
            mode=mode,
            marker=dict(size=marker_size) if mode in ['markers', 'lines+markers'] else None,
            line = line,
            showlegend=True if (row == 1 and col == 1) else False
        ),
        row=row, 
        col=col
    )

def add_data_and_psd(fig, time, data, name, row, col, data_id=0, mode='lines', dash="solid", marker_size=4):
    assert data_id < len(COLORS)
    line = {"dash": dash, "color": COLORS[data_id]}

    """Add time series and PSD data to plotly figure"""
    # Add time series data
    start = int(0.8*len(data))
    fig.add_trace(
        go.Scatter(
            x=time[start:], 
            y=data[start:], 
            name=name,
            legendgroup=name,
            mode=mode,
            marker=dict(size=marker_size) if mode in ['markers', 'lines+markers'] else None,
            line = line,
            showlegend=True if (row == 1 and col == 1) else False
        ),
        row=row, 
        col=col
    )

    # peaks_idx = find_peak_idx(data)
    # fig.add_trace(
    #     go.Scattergl(
    #         x=time[peaks_idx], 
    #         y=data[peaks_idx],
    #         mode='markers',
    #     ),
    #     row=row_data, 
    #     col=col_data
    # )

    # Add PSD data
    frequencies, psd = compute_psd(time, data)
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=psd,
            name=name,
            legendgroup=name,
            mode=mode,
            marker=dict(size=marker_size) if mode in ['markers', 'lines+markers'] else None,
            line = line,
            showlegend=False
        ),
        row=row+1,
        col=col
    )
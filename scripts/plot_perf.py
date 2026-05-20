import plotting as plot
import plotly.graph_objects as go

def plot_fig(prefix, meshes, cpu_times, gpu_times):
    # Time comparison
    fig = plot.fig_create_multi(1,1)
    fig.add_trace(
        go.Scatter(
            x=meshes,
            y=cpu_times,
            name="CPU",
            mode='lines+markers',
            marker=dict(size=6),
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=meshes,
            y=gpu_times,
            name="GPU",
            mode='lines+markers',
            marker=dict(size=6),
        ),
        row=1,
        col=1
    )
    plot.format_subplot(fig, 1, 1, "Mesh Size (panels)", "Computation Time (s)", ".0e")
    fig.update_yaxes(type="log", dtick=1, row=1, col=1)
    plot.fig_save(fig, f"build/{prefix}_perf_compare")

    # Speedup comparison
    speedups = [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]
    fig = plot.fig_create_multi(1,1)
    fig.add_trace(
        go.Scatter(
            x=meshes,
            y=speedups,
            name="Speedup (CPU / GPU)",
            mode='lines+markers',
            marker=dict(size=6),
        ),
        row=1,
        col=1
    )
    plot.format_subplot(fig, 1, 1, "Mesh Size (panels)", "GPU Speedup Factor", ".1f")
    plot.fig_save(fig, f"build/{prefix}_speedup_compare")

def plot_scaling(cores, times):
    fig = plot.fig_create_multi(1,1)
    fig.add_trace(
        go.Scatter(
            x=cores,
            y=times,
            name="Real",
            mode='lines+markers',
            marker=dict(size=6),
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=cores,
            y=cores,
            name="Ideal",
            mode='lines',
            line=dict(dash='dash', color='gray'),
            marker=dict(size=6),
        ),
        row=1,
        col=1
    )
    
    plot.format_subplot(fig, 1, 1, "CPU Cores", "Speedup Factor", ".0f")
    plot.fig_save(fig, "build/uvlm_cpu_core_scaling")

if __name__ == "__main__":
    uvlm_meshes = [500, 1000, 2000, 4000, 8000, 16000]
    uvlm_cpu = [0.17646, 0.89409, 7.28, 61.54, 513.59, 4216.68]
    # uvlm_cpu_sse2 = [0.202, 1.39, 11.8, 101.09, 872.0, 7251.0]
    uvlm_gpu = [0.07281, 0.19687, 1.40, 9.79, 74.89, 606.67]

    plot_fig("uvlm", uvlm_meshes, uvlm_cpu, uvlm_gpu)

    hbvlm_meshes = [500, 1000, 2000, 4000, 8000]
    hbvlm_cpu = [0.86011, 3.99, 19.50, 101.49, 403.94]
    hbvlm_gpu = [0.31037, 0.75419, 3.57, 15.34, 59.7]
    plot_fig("hbvlm", hbvlm_meshes, hbvlm_cpu, hbvlm_gpu)

    # uvlm_core_time = [61.95, 31.0, 16.4, 9.5]
    # core_speedups = [uvlm_core_time[0]/t for t in uvlm_core_time]
    # plot_scaling([1, 2, 4, 8], core_speedups)
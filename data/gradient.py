import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def create_mesh(ni, nj, ri=0.2, ro=1.0):
    """
    ni: number of cells radially
    nj: number of cells longitudinally
    """

    theta_vec = np.linspace(0, 2*np.pi, ni+1)
    r_vec = np.linspace(ri, ro, nj+1)

    verts = np.zeros((ni+1, nj+1, 2))
    centroids = np.zeros((ni, nj, 2))
    areas = np.zeros((ni, nj))

    for j, r in enumerate(r_vec):
        for i, theta in enumerate(theta_vec):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            verts[i, j, 0] = x
            verts[i, j, 1] = y

    for j in range(nj):
        for i in range(ni):
            v0 = verts[i, j]
            v1 = verts[i+1, j]
            v2 = verts[i+1, j+1]
            v3 = verts[i, j+1]
            centroids[i, j] = 0.25 * (v0 + v1 + v2 + v3)
            f = v3 - v0
            b = v2 - v0
            e = v1 - v0
            areas[i, j] = 0.5 * (np.linalg.norm(np.cross(f, b)) + np.linalg.norm(np.cross(b, e)))

    return verts, centroids, areas

# def f(x, y): return (3*x**2*y - y**3) / (x**2 + y**2)**1.5 + x**2 + y**2
# def dfdx(x, y):  return 2*x + (3*x*y*(3*y**2-x**2)) / (x**2 + y**2)**2.5
# def dfdy(x, y):  return 2*y + (3*x*x*(x**2-3*y**2)) / (x**2 + y**2)**2.5

def f(x, y): return x**2 + y**2
def dfdx(x, y): return 2*x
def dfdy(x, y): return 2*y

def f_(pair): return f(pair[0], pair[1])
def dfdx_(pair): return dfdx(pair[0], pair[1])
def dfdy_(pair): return dfdy(pair[0], pair[1])

rot_mat = np.array([[0, -1], [1, 0]])

if __name__ == "__main__":
    ni_vec = [10, 20, 40, 80, 160, 320]
    nj_vec = [5, 5, 5, 5, 5, 5]
    
    delta_r = []
    errors = []

    for ni, nj in zip(ni_vec, nj_vec):
        print(f"Mesh: {ni} x {nj}")
        verts, centroids, areas = create_mesh(ni, nj)
        
        grads = np.zeros((ni, nj, 2))
        analytical_grads = np.zeros((ni, nj, 2))

        vals = f(centroids[..., 0], centroids[..., 1])
        analytical_grads[..., 0] = dfdx(centroids[..., 0], centroids[..., 1])
        analytical_grads[..., 1] = dfdy(centroids[..., 0], centroids[..., 1])

        # Create cell polygons for plotting
        polygons = []
        for j in range(nj):
            for i in range(ni):
                poly = np.array([
                    verts[i, j],
                    verts[i+1, j],
                    verts[i+1, j+1],
                    verts[i, j+1]
                ])
                polygons.append(poly)

        # Green-Gauss method
        for j in range(1, nj-1):
            for i in range(1, ni-1):
                grads[i, j] += f_(centroids[i, j-1]) * rot_mat @ (verts[i+1, j] - verts[i, j])
                grads[i, j] += f_(centroids[i+1, j]) * rot_mat @ (verts[i+1, j+1] - verts[i+1, j])
                grads[i, j] += f_(centroids[i, j+1]) * rot_mat @ (verts[i, j+1] - verts[i+1, j+1])
                grads[i, j] += f_(centroids[i-1, j]) * rot_mat @ (verts[i, j] - verts[i, j+1])
                grads[i, j] /= areas[i, j]

        l2_err = np.sqrt(np.sum(np.linalg.norm(grads[1:-1, 1:-1] - analytical_grads[1:-1, 1:-1], axis=2)**2 * areas[1:-1, 1:-1]) / np.sum(areas[1:-1, 1:-1]))
        
        delta_r.append(2*np.pi / ni)
        errors.append(l2_err)

        # Plot mesh and values for the first case only
        if ni == ni_vec[0]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot mesh colored by function values
            pc = PolyCollection(polygons, cmap='viridis')
            pc.set_array(vals.flatten())
            ax1.add_collection(pc)
            ax1.autoscale()
            ax1.set_aspect('equal')
            ax1.set_title('Function Values')
            fig.colorbar(pc, ax=ax1)

            # Plot mesh colored by error magnitude
            error_mag = np.linalg.norm(grads - analytical_grads, axis=2)
            pc2 = PolyCollection(polygons, cmap='viridis')
            pc2.set_array(error_mag.flatten())
            ax2.add_collection(pc2)
            ax2.autoscale()
            ax2.set_aspect('equal')
            ax2.set_title('Error Magnitude')
            fig.colorbar(pc2, ax=ax2)
            
            plt.show()

    # Convergence plot
    plt.figure(figsize=(8, 6))
    plt.loglog(delta_r, errors, 'o-', label='Green-Gauss')
    
    slope_ref = 2
    x_ref = np.array([min(delta_r), max(delta_r)])
    y_ref = errors[0] * (x_ref/delta_r[0])**slope_ref
    plt.loglog(x_ref, y_ref, '--', label=f'Reference slope ({slope_ref})')

    plt.grid(True)
    plt.xlabel('Δr')
    plt.ylabel('L2 Error')
    plt.title('Convergence Plot')
    plt.legend()
    plt.show()




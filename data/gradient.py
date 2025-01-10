import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

import scipy as sp

def mesh_metrics(verts):
    ni = verts.shape[0] - 1
    nj = verts.shape[1] - 1
    centroids = np.zeros((ni, nj, 2))
    areas = np.zeros((ni, nj))

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

    return centroids, areas

def create_semicylinder_mesh(ni, nj, ri=0.2, ro=1.0):
    """
    Creates a structured semicylinder mesh centered at 0,0
    Mesh starts at the inner radius and goes ouward, radially moves in clockwise direction
    ni: number of cells radially
    nj: number of cells longitudinally
    """

    theta_vec = np.linspace(np.pi, 0, ni+1) # clockwise
    r_vec = np.linspace(ri, ro, nj+1)

    verts = np.zeros((ni+1, nj+1, 2))

    for j, r in enumerate(r_vec):
        for i, theta in enumerate(theta_vec):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            verts[i, j, 0] = x
            verts[i, j, 1] = y

    return verts

def create_rectangle_mesh(ni, nj, w=1.0, h=1.0):
    """
    Creates a structured rectangular mesh centered at 0,0
    Mesh starts in the bottom left corner
    ni: number of cells in x direction
    nj: number of cells in y direction
    w: width of the mesh
    h: height of the mesh

    """

    x_vec = np.linspace(-0.5*w, 0.5*w, ni+1)
    y_vec = np.linspace(-0.5*h, 0.5*h, nj+1)

    verts = np.zeros((ni+1, nj+1, 2))

    for j, y in enumerate(y_vec):
        for i, x in enumerate(x_vec):
            verts[i, j, 0] = x
            verts[i, j, 1] = y

    return verts

def f(x, y): return (3*x**2*y - y**3) / (x**2 + y**2)**1.5 + x**2 + y**2
def dfdx(x, y):  return 2*x + (3*x*y*(3*y**2-x**2)) / (x**2 + y**2)**2.5
def dfdy(x, y):  return 2*y + (3*x*x*(x**2-3*y**2)) / (x**2 + y**2)**2.5

# def f(x, y): return y**2 / (x**2 + y**2)
# def dfdx(x, y): return - 2*y**2*x / (x**2 + y**2)**2
# def dfdy(x, y): return 2*y*x**2 / (x**2 + y**2)**2

# def f(x, y): return x**2 + y**2
# def dfdx(x, y): return 2*x
# def dfdy(x, y): return 2*y

# def f(x, y): return x + y
# def dfdx(x, y): return 1
# def dfdy(x, y): return 1

def f_(pair): return f(pair[0], pair[1])
def dfdx_(pair): return dfdx(pair[0], pair[1])
def dfdy_(pair): return dfdy(pair[0], pair[1])

rot_mat_cc = np.array([[0, -1], [1, 0]]) # counter-clockwise rotation matrix
rot_mat_c = np.array([[0, 1], [-1, 0]]) # clockwise rotation matrix

def gg(f: callable, centroids, areas, verts):
    ni, nj = centroids.shape[:2]
    grads = np.zeros((ni, nj, 2))
    # Green-Gauss method
    for j in range(1, nj-1):
        for i in range(1, ni-1):
            f0 = f_(centroids[i, j])
            grads[i, j] += 0.5 * (f_(centroids[i, j-1]) + f0) * rot_mat_c @ (verts[i+1, j] - verts[i, j])
            grads[i, j] += 0.5 * (f_(centroids[i+1, j]) + f0) * rot_mat_c @ (verts[i+1, j+1] - verts[i+1, j])
            grads[i, j] += 0.5 * (f_(centroids[i, j+1]) + f0) * rot_mat_c @ (verts[i, j+1] - verts[i+1, j+1])
            grads[i, j] += 0.5 * (f_(centroids[i-1, j]) + f0) * rot_mat_c @ (verts[i, j] - verts[i, j+1])
            grads[i, j] /= areas[i, j]

    return grads

def mgg(f: callable, centroids, areas, verts, tol=1e-8, max_iter=10):
    ni, nj = centroids.shape[:2]
    grads = np.zeros((ni, nj, 2))
    new_grads = np.zeros((ni, nj, 2))

    def mgg_face_contribution(v0, v1, c0, c1, f0, f1, g0, g1):
        x_f = 0.5 * (v1 + v0)
        delta_s = np.linalg.norm(v1 - v0)
        normal = rot_mat_c @ (v1 - v0) / delta_s
        delta_r = np.linalg.norm(c1 - c0)
        r_f = (c1 - c0) / delta_r
        alpha = np.dot(normal, r_f)
        grad_n = alpha * (f1 - f0) / delta_r + 0.5 * np.dot(g0 + g1, normal - alpha * r_f)
        return grad_n * (x_f - c0) * delta_s

    delta = 1
    iteration = 0
    while delta > tol and iteration < max_iter:
        delta = 0
        for j in range(1, nj-1):
            for i in range(1, ni-1):
                f0 = f_(centroids[i, j])
                v0 = verts[i, j]
                v1 = verts[i+1, j]
                v2 = verts[i+1, j+1]
                v3 = verts[i, j+1]
                grad = np.zeros(2)
                grad += mgg_face_contribution(v0, v1, centroids[i, j], centroids[i, j-1], f0, f_(centroids[i, j-1]), new_grads[i, j], new_grads[i, j-1])
                grad += mgg_face_contribution(v1, v2, centroids[i, j], centroids[i+1, j], f0, f_(centroids[i+1, j]), new_grads[i, j], new_grads[i+1, j])
                grad += mgg_face_contribution(v2, v3, centroids[i, j], centroids[i, j+1], f0, f_(centroids[i, j+1]), new_grads[i, j], new_grads[i, j+1])
                grad += mgg_face_contribution(v3, v0, centroids[i, j], centroids[i-1, j], f0, f_(centroids[i-1, j]), new_grads[i, j], new_grads[i-1, j])

                grad /= areas[i, j]
                delta += np.sum((grad - grads[i, j])**2)
                grads[i, j] = grad

        new_grads = grads.copy()
        delta = np.sqrt(delta)
        iteration += 1
    print(f"MGG Iterations: {iteration}")

    return grads

def rbf_gradients(f: callable, centroids, areas, verts, epsilon=1.0):
    ni, nj = centroids.shape[:2]
    grads = np.zeros((ni, nj, 2))
    
    def gaussian_rbf(r, eps=epsilon):
        return np.exp(-(eps*r)**2)
    
    def gaussian_rbf_dx(x, y, x0, y0, eps=epsilon):
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        return -2*eps**2*(x-x0)*gaussian_rbf(r, eps)
    
    def gaussian_rbf_dy(x, y, x0, y0, eps=epsilon):
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        return -2*eps**2*(y-y0)*gaussian_rbf(r, eps)

    # For each interior point
    for j in range(1, nj-1):
        for i in range(1, ni-1):
            # Collect stencil points (including point itself)
            stencil_i = []
            stencil_j = []
            stencil_f = []
            
            # Add neighbors in a wider stencil
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < ni and 0 <= jj < nj:  # check both bounds
                        stencil_i.append(centroids[ii, jj, 0])
                        stencil_j.append(centroids[ii, jj, 1])
                        stencil_f.append(f_(centroids[ii, jj]))
            
            stencil_i = np.array(stencil_i)
            stencil_j = np.array(stencil_j)
            stencil_f = np.array(stencil_f)
            
            # Center point
            x0 = centroids[i, j, 0]
            y0 = centroids[i, j, 1]
            
            # Build RBF interpolation matrix
            n_points = len(stencil_i)
            A = np.zeros((n_points, n_points))
            for k in range(n_points):
                for l in range(n_points):
                    r = np.sqrt((stencil_i[k]-stencil_i[l])**2 + 
                              (stencil_j[k]-stencil_j[l])**2)
                    A[k, l] = gaussian_rbf(r)
            
            # Solve for RBF coefficients
            coeffs = np.linalg.solve(A, stencil_f)
            
            # Evaluate derivatives at center point
            dfdx = 0
            dfdy = 0
            for k in range(n_points):
                dfdx += coeffs[k] * gaussian_rbf_dx(x0, y0, stencil_i[k], stencil_j[k])
                dfdy += coeffs[k] * gaussian_rbf_dy(x0, y0, stencil_i[k], stencil_j[k])
            
            grads[i, j] = [dfdx, dfdy]
    
    return grads

def calculate_l2_error(grads, analytical_grads, areas):
    """Calculate L2 error excluding boundary cells"""
    diff = np.zeros_like(grads)
    diff[1:-1, 1:-1] = grads[1:-1, 1:-1] - analytical_grads[1:-1, 1:-1]
    return np.sqrt(np.sum(np.sum(diff[1:-1, 1:-1]**2, axis=2) * areas[1:-1, 1:-1]) / np.sum(areas[1:-1, 1:-1]))

def plot_results(verts, vals, grads, analytical_grads, title=""):
    """Plot mesh colored by function values and error magnitude"""
    ni, nj = vals.shape
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

    diff = np.zeros_like(grads)
    diff[1:-1, 1:-1] = grads[1:-1, 1:-1] - analytical_grads[1:-1, 1:-1]
    error_mag = np.linalg.norm(diff, axis=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot mesh colored by function values
    pc = PolyCollection(polygons, cmap='viridis')
    pc.set_array(vals.flatten(order="F"))
    ax1.add_collection(pc)
    ax1.autoscale()
    ax1.set_aspect('equal')
    ax1.set_title(f'{title} - Function Values')
    fig.colorbar(pc, ax=ax1)

    # Colored by error magnitude
    pc2 = PolyCollection(polygons, cmap='viridis', norm=Normalize(vmin=0, vmax=1))
    pc2.set_array(error_mag.flatten(order="F"))
    ax2.add_collection(pc2)
    ax2.autoscale()
    ax2.set_aspect('equal')
    ax2.set_title(f'{title} - Error Magnitude')
    fig.colorbar(pc2, ax=ax2)
    
    plt.show()

if __name__ == "__main__":
    ni_vec = [8, 16, 32, 64, 128, 256, 512]
    nj_vec = [8, 16, 32, 64, 128, 256, 512]
    
    # Define methods to compare
    gradient_methods = [
        ("Green-Gauss", gg),
        ("Modified Green-Gauss", mgg),
        ("RBF", rbf_gradients)
    ]
    
    delta_r = [(ni*nj)**(-1/2) for ni, nj in zip(ni_vec, nj_vec)]
    errors = {name: [] for name, _ in gradient_methods}

    for ni, nj in zip(ni_vec, nj_vec):
        print(f"Mesh: {ni} x {nj}")
        verts = create_semicylinder_mesh(ni, nj)
        centroids, areas = mesh_metrics(verts)        
        
        # Calculate analytical solution
        vals = f(centroids[..., 0], centroids[..., 1])
        analytical_grads = np.zeros((ni, nj, 2))
        analytical_grads[..., 0] = dfdx(centroids[..., 0], centroids[..., 1])
        analytical_grads[..., 1] = dfdy(centroids[..., 0], centroids[..., 1])

        # Compute gradients using all methods
        for method_name, method_func in gradient_methods:
            grads = method_func(f_, centroids, areas, verts)
            l2_err = calculate_l2_error(grads, analytical_grads, areas)
            errors[method_name].append(l2_err)

            # Plot first case only
            if ni == ni_vec[1]:
                plot_results(verts, vals, grads, analytical_grads, method_name)

    # Convergence plot
    plt.figure(figsize=(8, 6))
    
    for method_name, _ in gradient_methods:
        plt.loglog(delta_r, errors[method_name], 'o-', label=method_name)
    
    x_ref = np.array([delta_r[0], delta_r[1]])
    y_ref_2 = (x_ref/delta_r[0])**2 * errors[list(errors.keys())[0]][0]
    y_ref_1 = (x_ref/delta_r[0])**1 * errors[list(errors.keys())[0]][0]
    plt.loglog(x_ref, y_ref_1, '--', label='O(h^1)')
    plt.loglog(x_ref, y_ref_2, '--', label='O(h^2)')

    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.xlabel('Δr')
    plt.ylabel('L2 Error')
    plt.title('Convergence Plot')
    plt.grid(True)
    plt.legend()
    plt.show()


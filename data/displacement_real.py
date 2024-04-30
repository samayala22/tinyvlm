import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def coords_to_verts(coords, nc, ns):
    assert coords.shape == (3, (ns+1)*(nc+1))
    verts = []
    if (nc > 0): 
        for ia in range(nc*ns):
            v0 = ia + ia // ns
            v1 = v0 + 1
            v3 = v0 + ns+1
            v2 = v3 + 1
            idx = [v0, v1, v2, v3]
            verts.append([[coords[0, i], coords[1, i], coords[2, i]] for i in idx])

    return verts

coords_wing = []
coords_wake = []

with open("build/windows/x64/debug/wing_data.txt") as f:
    # first line
    nc, ns = map(int, f.readline().split())
    timesteps = int(f.readline())
    for i in range(timesteps):
        coords = np.zeros((3, (ns+1)*(nc+1)), dtype=np.float32)
        f.readline() # skip newline
        coords[0, :] = np.array(list(map(float, f.readline().split())))
        coords[1, :] = np.array(list(map(float, f.readline().split())))
        coords[2, :] = np.array(list(map(float, f.readline().split())))
        coords_wing.append(coords)

with open("build/windows/x64/debug/wake_data.txt") as f:
    # first line
    f.readline() # skip first line
    f.readline() # skip second line
    for i in range(timesteps):
        coords = np.zeros((3, (ns+1)*(i+1)), dtype=np.float32)
        f.readline() # skip newline
        coords[0, :] = np.array(list(map(float, f.readline().split())))
        coords[1, :] = np.array(list(map(float, f.readline().split())))
        coords[2, :] = np.array(list(map(float, f.readline().split())))
        coords_wake.append(coords)

# # Setup animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Setting the initial view to be isometric
ax.view_init(elev=30, azim=135)  # Isometric view angle
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(-30, 0)
ax.set_ylim(-15, 15)
ax.set_zlim(-15, 15)
ax.invert_xaxis()  # Invert x axis
ax.invert_yaxis()  # Invert y axis

poly_wing = Poly3DCollection(coords_to_verts(coords_wing[0], nc, ns), facecolors=['b'], edgecolor='k')
poly_wake = Poly3DCollection(coords_to_verts(coords_wake[0], 0, ns), facecolors=['r'], edgecolor='k')
ax.add_collection3d(poly_wing)
ax.add_collection3d(poly_wake)

def update(frame, poly_wing, poly_wake):
    global coords_wing, coords_wake

    poly_wing.set_verts(coords_to_verts(coords_wing[frame], nc, ns))
    poly_wake.set_verts(coords_to_verts(coords_wake[frame], frame, ns))

    return poly_wing, poly_wake

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, timesteps), fargs=(poly_wing, poly_wake), blit=False, repeat=False)
# ani.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

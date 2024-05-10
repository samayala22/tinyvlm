import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

class Kinematics:
    joints = [] # list of transformation lambdas

    def add_joint(self, joint):
        self.joints.append(joint)

    def displacement(self, t):
        disp = np.identity(4)
        for i in range(len(self.joints)):
            disp = disp @ self.joints[i](t)
        return disp

    def relative_displacement(self, t0, t1):
        return self.displacement(t1) @ np.linalg.inv(self.displacement(t0))

    def velocity(self, t, vertex):
        EPS_sqrt_f = np.sqrt(1.19209e-07)
        return(self.relative_displacement(t, t+EPS_sqrt_f) @ vertex - vertex) / EPS_sqrt_f

    def absolute_velocity(self, t, vertex):
        return np.linalg.norm(self.velocity(t, vertex))
    
def skew_matrix(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def translation_matrix(v):
    return np.array([[1, 0, 0, v[0]],
                     [0, 1, 0, v[1]],
                     [0, 0, 1, v[2]],
                     [0, 0, 0, 1]])

def rotation_matrix(center, axis, theta):
    axis = axis[:3]
    center = center[:3]
    mat = np.identity(4)
    rodrigues = np.identity(3) + np.sin(theta) * skew_matrix(axis) + (1 - np.cos(theta)) * skew_matrix(axis) @ skew_matrix(axis)
    mat[:3, :3] = rodrigues
    mat[:3, 3] = (np.identity(3) - rodrigues) @ center # matvec
    return mat

def move_vertices(vertices, displacement):
    return displacement @ vertices


# def displacement_rotor(t, frame): 
#     return rotation_matrix(frame @ [0, 0, 0, 1], frame @ [0, 0, 1, 0], 1 * t)
# def displacement_rotor(t): 
#     return rotation_matrix([0, 0, 0], [0, 0, 1], 7 * t)
def displacement_wing(t): return translation_matrix([0, 0, np.sin(0.9 * t)])
def freestream(t): return translation_matrix([-1 * t, 0, 0])
alpha = np.radians(5)
def freestream2(t): return translation_matrix([-np.cos(alpha)*t, 0, -np.sin(alpha)*t])
def pitching(t): 
    return rotation_matrix([0, 0, 0], [0, 1, 0], 0.25 * np.pi * np.sin(t))

kinematics = Kinematics()
kinematics.add_joint(freestream2)
kinematics.add_joint(pitching)

dt = 0.2
t_final = 15

# vertices of a single panel (clockwise) (initial position) (global coordinates)
vertices = np.array([
    [0.0, 2.0, 4.0, 4.0], # x
    [0.0, 10.0, 10.0, 0.0], # y
    [0.0, 0.0, 0.0, 0.0], # z
    [1.0, 1.0, 1.0, 1.0]  # homogeneous coordinates
])
vertices = np.column_stack((vertices, vertices[:,0]))

# print("Analytical vel: ", [-np.cos(alpha), 0, -np.sin(alpha)])
print("Numerical vel: ", kinematics.velocity(0, [4.0,0,0,1]))

# Setup animation
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

# Initial plot - create a line object
line, = ax.plot3D(vertices[0, :], vertices[1, :], vertices[2, :], '-') 
scatter = ax.scatter(vertices[0, :], vertices[1, :], vertices[2, :], c='r', marker='o')

current_frame = 0
def update(frame):
    global vertices, kinematics, current_frame
    t = frame * dt
    
    vertices_velocity = np.array([kinematics.absolute_velocity(t, vertex) for vertex in vertices.T])
    # if frame == current_frame: # otherwise invalid velocity value
    #     print(f"velocity: {kinematics.velocity(t, vertices[:, 0])}")
    #     print(f"frame: {frame} | vel: {vertices_velocity[:-1]}")

    norm = plt.Normalize(vertices_velocity.min(), vertices_velocity.max())
    colors = cm.viridis(norm(vertices_velocity))

    # Update the line object for 3D
    line.set_data(vertices[0, :], vertices[1, :])  # y and z for 2D part of set_data
    line.set_3d_properties(vertices[2, :])  # x for the 3rd dimension

    scatter._offsets3d = (vertices[0, :], vertices[1, :], vertices[2, :])
    scatter.set_facecolor(colors)

    if (frame == current_frame): # fix double frame 0 issue
        print(f"t = {t:.2f}/{t_final}", end='\r')
        vertices = move_vertices(vertices, kinematics.relative_displacement(t, t+dt))
        current_frame += 1

    return line, scatter

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, t_final/dt), blit=False, repeat=False)
# ani.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

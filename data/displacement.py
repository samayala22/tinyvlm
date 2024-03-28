import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Kinematics:
    joints = [] # list of transformation lambdas

    def add_joint(self, joint, frame):
        self.joints.append(joint)

    def displacement(self, t):
        disp = np.identity(4)
        for i in range(len(self.joints)):
            disp = disp @ self.joints[i](t)
        return disp

    def relative_displacement(self, t0, t1):
        return self.displacement(t1) @ np.linalg.inv(self.displacement(t0))

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

def rotation_matrix2(axis, theta):
    axis = np.array(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c), 0],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b), 0],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c, 0],
                     [0, 0, 0, 1]])

def move_vertices(vertices, displacement):
    return displacement @ vertices

def displacement_wing(t): return translation_matrix([0, 0, np.sin(0.9 * t)])
def displacement_freestream(t): return translation_matrix([-2 * t, 0, 0])
# def displacement_rotor(t, frame): 
#     return rotation_matrix(frame @ [0, 0, 0, 1], frame @ [0, 0, 1, 0], 1 * t)
def displacement_rotor(t): 
    return rotation_matrix([0, 0, 0], [0, 0, 1], 7 * t)
# def displacement_rotor(t, frame):
#     return rotation_matrix2([0, 0, 1], 1 * t)

kinematics = Kinematics()
kinematics.add_joint(displacement_freestream, np.identity(4))
kinematics.add_joint(displacement_wing, np.identity(4))
kinematics.add_joint(displacement_rotor, np.identity(4))

dt = 0.01
t_final = 20

# vertices of a single panel (clockwise) (initial position) (global coordinates)
vertices = np.array([
    [0.0, 2.0, 4.0, 4.0], # x
    [0.0, 10.0, 10.0, 0.0], # y
    [0.0, 0.0, 0.0, 0.0], # z
    [1.0, 1.0, 1.0, 1.0]  # homogeneous coordinates
])
vertices = np.column_stack((vertices, vertices[:,0]))

body_frame = np.identity(4)


wing_frame = np.identity(4)

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
ax.set_zlim(-5, 5)
ax.invert_xaxis()  # Invert x axis
ax.invert_yaxis()  # Invert y axis

# Initial plot - create a line object
line, = ax.plot3D(vertices[0, :], vertices[1, :], vertices[2, :], 'o-') 

def update(frame):
    global vertices, kinematics
    t = frame * dt
    print(f"t = {t:.2f}/{t_final}", end='\r')
    vertices = move_vertices(vertices, kinematics.relative_displacement(t, t+dt))
    # Update the line object for 3D
    line.set_data(vertices[0, :], vertices[1, :])  # y and z for 2D part of set_data
    line.set_3d_properties(vertices[2, :])  # x for the 3rd dimension
    return line,

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, t_final/dt), blit=False, repeat=False)
# ani.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Kinematics:
    joints = [] # list of transformation lambdas
    frames = [np.identity(4)] # list of joint frames in global coordinates

    def add_joint(self, joint, frame):
        self.joints.append(joint)
        self.frames.append(frame)

    def displacement(self, t, move_frames=True):
        disp = np.identity(4)
        for i in range(len(self.joints)):
            disp = disp @ self.joints[i](t, self.frames[i])
            if move_frames: 
                self.frames[i+1] = self.joints[i](t, self.frames[i]) @ self.frames[i+1]
        return disp

    def relative_displacement(self, t0, t1):
        base = np.linalg.inv(self.displacement(t0, True))
        return self.displacement(t1, False) @ base

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

def displacement_wing(t, frame): return translation_matrix([0, 0, np.sin(0.9 * t)])
def displacement_freestream(t, frame): return translation_matrix([-1 * t, 0, 0])
def displacement_rotor(t, frame): return rotation_matrix(frame @ [0, 0, 0, 1], frame @ [0, 0, 1, 0], 1 * t)
#def displacement_rotor(t, frame): return rotation_matrix([0, 0, 0, 1], [0, 0, 1, 0], 1 * t)

kinematics = Kinematics()
kinematics.add_joint(displacement_freestream, np.identity(4))
kinematics.add_joint(displacement_wing, np.identity(4))
# kinematics.add_joint(displacement_rotor, np.identity(4))

dt = 0.5
t_final = 15

# vertices of a single panel (clockwise) (initial position) (global coordinates)
vertices = np.array([
    [0.0, 0.5, 1.0, 1.0], # x
    [0.0, 10.0, 10.0, 0.0], # y
    [0.0, 0.0, 0.0, 0.0], # z
    [1.0, 1.0, 1.0, 1.0]  # homogeneous coordinates
])

body_frame = np.identity(4)

vertices = np.column_stack((vertices, vertices[:,0]))

wing_frame = np.identity(4)

# Setup animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Setting the initial view to be isometric
ax.view_init(elev=30, azim=135)  # Isometric view angle
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(-15, 0)
ax.set_ylim(-15, 15)
ax.set_zlim(-5, 5)
ax.invert_xaxis()  # Invert x axis
ax.invert_yaxis()  # Invert y axis

# Initial plot - create a line object
line, = ax.plot3D(vertices[0, :], vertices[1, :], vertices[2, :], 'o-') 

def update(frame):
    t = frame * dt
    current_displacement = kinematics.relative_displacement(0, t)
    moved_vertices = move_vertices(vertices, current_displacement)
    # Update the line object for 3D
    line.set_data(moved_vertices[0, :], moved_vertices[1, :])  # y and z for 2D part of set_data
    line.set_3d_properties(moved_vertices[2, :])  # x for the 3rd dimension
    return line,

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, t_final/dt), blit=False)

plt.show()

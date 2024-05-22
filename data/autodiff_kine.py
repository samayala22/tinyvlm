from typing import Union
import math
import numpy as np

class Var:
    """
    A variable which holds a number and enables gradient computations.
    """

    def __init__(self, val: Union[float, int], parents=None):
        assert type(val) in {float, int}
        self.v = val
        self.parents = parents if parents is not None else []
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for parent, grad in self.parents:
            parent.backprop(grad * bp)

    def backward(self):
        self.zero_grad()  # Reset gradients before backpropagation
        self.backprop(1.0)

    def zero_grad(self):
        self.grad = 0.0
        for parent, grad in self.parents:
            parent.zero_grad()

    def __add__(self, other) -> 'Var':
        other = other if isinstance(other, Var) else Var(other)
        return Var(self.v + other.v, [(self, 1.0), (other, 1.0)])

    def __mul__(self, other) -> 'Var':
        other = other if isinstance(other, Var) else Var(other)
        return Var(self.v * other.v, [(self, other.v), (other, self.v)])

    def __pow__(self, power: Union[float, int]) -> 'Var':
        power = power if isinstance(power, Var) else Var(power)
        return Var(self.v ** power.v, [(self, power.v * self.v ** (power.v - 1)), (power, self.v ** power.v * math.log(self.v))])

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return "Var(v=%.4f, grad=%.4f)" % (self.v, self.grad)

def sin(var: Var) -> Var:
    return Var(math.sin(var.v), [(var, math.cos(var.v))])

def cos(var: Var) -> Var:
    return Var(math.cos(var.v), [(var, -math.sin(var.v))])

def tan(var: Var) -> Var:
    return Var(math.tan(var.v), [(var, 1 / math.cos(var.v) ** 2)])

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
kinematics.add_joint(freestream)
kinematics.add_joint(freestream2)
# kinematics.add_joint(pitching)
t = Var(0)
disp = kinematics.displacement(t)
vertex = np.array([0, 0, 0, 1])
pos = disp @ vertex

pos[0].backward()
print(t.grad)
pos[1].backward()
print(t.grad)
pos[2].backward()
print(t.grad)
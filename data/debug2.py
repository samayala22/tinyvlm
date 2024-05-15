import numpy as np

class DualQuaternion:
    def __init__(self, real, dual):
        self.real = real  # Quaternion
        self.dual = dual  # Quaternion

    @staticmethod
    def from_translation(translation):
        real = np.array([1, 0, 0, 0], dtype=translation.dtype)
        translation = np.array(translation, dtype=translation.dtype)
        dual = np.hstack(([0], 0.5 * translation))
        return DualQuaternion(real, dual)

    @staticmethod
    def from_rotation(axis_origin, axis, angle):
        axis = np.array(axis, dtype=angle.dtype)
        axis /= np.linalg.norm(axis)
        sin_half_angle = np.sin(angle / 2)
        real = np.hstack(([np.cos(angle / 2)], sin_half_angle * axis)).astype(angle.dtype)

        origin = np.array(axis_origin, dtype=angle.dtype)
        dual_part = np.cross(-origin, sin_half_angle * axis)
        dual = np.hstack(([0], dual_part)).astype(angle.dtype)

        return DualQuaternion(real, dual)

    def transform_point(self, point):
        p = np.array([0, point[0], point[1], point[2]], dtype=self.real.dtype)
        transformed_p = self._quaternion_mul(self._quaternion_mul(self.real, p), self._quaternion_conj(self.real)) + 2 * self._quaternion_mul(self.dual, self._quaternion_conj(self.real))
        return transformed_p[1:4]

    def __mul__(self, other):
        real = self._quaternion_mul(self.real, other.real)
        dual = self._quaternion_mul(self.real, other.dual) + self._quaternion_mul(self.dual, other.real)
        return DualQuaternion(real, dual)

    @staticmethod
    def _quaternion_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z], dtype=q1.dtype)

    @staticmethod
    def _quaternion_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)

alpha = np.radians(5)  # Example angle in radians

def freestream2(t):
    return DualQuaternion.from_translation([-np.cos(alpha) * t, 0, -np.sin(alpha) * t])

def pitching(t):
    return DualQuaternion.from_rotation([0, 0, 0], [0, 1, 0], 0.25 * np.pi * np.sin(t))

class Kinematics:
    def __init__(self):
        self.m_joints = []

    def add_joint(self, joint):
        self.m_joints.append(joint)

    def displacement(self, t, n=0):
        result = DualQuaternion(np.array([1, 0, 0, 0], dtype=t.dtype), np.array([0, 0, 0, 0], dtype=t.dtype))
        end_joint = len(self.m_joints) if n == 0 else n
        for i in range(end_joint):
            result = result * self.m_joints[i](t)
        return result

    def relative_displacement(self, t0, t1, n=0):
        displacement_t1 = self.displacement(t1, n)
        displacement_t0 = self.displacement(t0, n)
        displacement_t0_inv = DualQuaternion(displacement_t0.real, -displacement_t0.dual)
        return displacement_t1 * displacement_t0_inv

    def velocity(self, t, vertex, n=0):
        EPS = np.sqrt(np.finfo(np.float32).eps)
        complex_t = t + 1j * EPS
        displacement_with_complex_step = self.relative_displacement(t, complex_t, n)
        vertex_complex = np.array(vertex, dtype=np.complex64)
        transformed_vertex = displacement_with_complex_step.transform_point(vertex_complex)
        return transformed_vertex.imag / EPS

    def velocity_magnitude(self, t, vertex):
        return np.linalg.norm(self.velocity(t, vertex))

# Example usage
kinematics = Kinematics()
kinematics.add_joint(freestream2)
kinematics.add_joint(pitching)

vertex = np.array([4.0, 0, 0], dtype=np.float32)  # Vertex without homogeneous coordinate
t = 0.0
velocity = kinematics.velocity(t, vertex)
velocity_magnitude = kinematics.velocity_magnitude(t, vertex)
print("Numerical vel: ", velocity)
print("Velocity Magnitude:", velocity_magnitude)

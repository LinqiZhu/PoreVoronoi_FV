# Generated from the unchanged Manufactured symbolic class.
import numpy as np
from numpy import sin, cos, pi

def L16_u0(x, y, z):
    return (1/8)*pi*sin((1/16)*pi*x)**2*sin((1/16)*pi*y)*sin((1/16)*pi*z)**2*cos((1/16)*pi*y)

def L16_u1(x, y, z):
    return -1/8*pi*sin((1/16)*pi*x)*sin((1/16)*pi*y)**2*sin((1/16)*pi*z)**2*cos((1/16)*pi*x)

def L16_u2(x, y, z):
    return 0

def L16_p0(x, y, z):
    return sin((1/8)*pi*x)*sin((1/8)*pi*y)*sin((1/8)*pi*z)

def L16_f0(x, y, z):
    return 0.005859375*pi**3*sin((1/16)*pi*x)**2*sin((1/16)*pi*y)*sin((1/16)*pi*z)**2*cos((1/16)*pi*y) - 0.0009765625*pi**3*sin((1/16)*pi*x)**2*sin((1/16)*pi*y)*cos((1/16)*pi*y) - 0.0009765625*pi**3*sin((1/16)*pi*y)*sin((1/16)*pi*z)**2*cos((1/16)*pi*y) + 0.125*pi*sin((1/8)*pi*y)*sin((1/8)*pi*z)*cos((1/8)*pi*x)

def L16_f1(x, y, z):
    return -0.005859375*pi**3*sin((1/16)*pi*x)*sin((1/16)*pi*y)**2*sin((1/16)*pi*z)**2*cos((1/16)*pi*x) + 0.0009765625*pi**3*sin((1/16)*pi*x)*sin((1/16)*pi*y)**2*cos((1/16)*pi*x) + 0.0009765625*pi**3*sin((1/16)*pi*x)*sin((1/16)*pi*z)**2*cos((1/16)*pi*x) + 0.125*pi*sin((1/8)*pi*x)*sin((1/8)*pi*z)*cos((1/8)*pi*y)

def L16_f2(x, y, z):
    return (1/8)*pi*sin((1/8)*pi*x)*sin((1/8)*pi*y)*cos((1/8)*pi*z)

def L24_u0(x, y, z):
    return 0.0833333333333333*pi*sin(0.0416666666666667*pi*x)**2*sin(0.0416666666666667*pi*y)*sin(0.0416666666666667*pi*z)**2*cos(0.0416666666666667*pi*y)

def L24_u1(x, y, z):
    return -0.0833333333333333*pi*sin(0.0416666666666667*pi*x)*sin(0.0416666666666667*pi*y)**2*sin(0.0416666666666667*pi*z)**2*cos(0.0416666666666667*pi*x)

def L24_u2(x, y, z):
    return 0

def L24_p0(x, y, z):
    return sin(0.0833333333333333*pi*x)*sin(0.0833333333333333*pi*y)*sin(0.0833333333333333*pi*z)

def L24_f0(x, y, z):
    return pi*(0.00173611111111111*pi**2*sin(0.0416666666666667*pi*x)**2*sin(0.0416666666666667*pi*y)*sin(0.0416666666666667*pi*z)**2*cos(0.0416666666666667*pi*y) - 0.000289351851851852*pi**2*sin(0.0416666666666667*pi*x)**2*sin(0.0416666666666667*pi*y)*cos(0.0416666666666667*pi*y) - 0.000289351851851852*pi**2*sin(0.0416666666666667*pi*y)*sin(0.0416666666666667*pi*z)**2*cos(0.0416666666666667*pi*y) + 0.0833333333333333*sin(0.0833333333333333*pi*y)*sin(0.0833333333333333*pi*z)*cos(0.0833333333333333*pi*x))

def L24_f1(x, y, z):
    return pi*(-0.00173611111111111*pi**2*sin(0.0416666666666667*pi*x)*sin(0.0416666666666667*pi*y)**2*sin(0.0416666666666667*pi*z)**2*cos(0.0416666666666667*pi*x) + 0.000289351851851852*pi**2*sin(0.0416666666666667*pi*x)*sin(0.0416666666666667*pi*y)**2*cos(0.0416666666666667*pi*x) + 0.000289351851851852*pi**2*sin(0.0416666666666667*pi*x)*sin(0.0416666666666667*pi*z)**2*cos(0.0416666666666667*pi*x) + 0.0833333333333333*sin(0.0833333333333333*pi*x)*sin(0.0833333333333333*pi*z)*cos(0.0833333333333333*pi*y))

def L24_f2(x, y, z):
    return 0.0833333333333333*pi*sin(0.0833333333333333*pi*x)*sin(0.0833333333333333*pi*y)*cos(0.0833333333333333*pi*z)

def L32_u0(x, y, z):
    return (1/16)*pi*sin((1/32)*pi*x)**2*sin((1/32)*pi*y)*sin((1/32)*pi*z)**2*cos((1/32)*pi*y)

def L32_u1(x, y, z):
    return -1/16*pi*sin((1/32)*pi*x)*sin((1/32)*pi*y)**2*sin((1/32)*pi*z)**2*cos((1/32)*pi*x)

def L32_u2(x, y, z):
    return 0

def L32_p0(x, y, z):
    return sin((1/16)*pi*x)*sin((1/16)*pi*y)*sin((1/16)*pi*z)

def L32_f0(x, y, z):
    return 0.000732421875*pi**3*sin((1/32)*pi*x)**2*sin((1/32)*pi*y)*sin((1/32)*pi*z)**2*cos((1/32)*pi*y) - 0.0001220703125*pi**3*sin((1/32)*pi*x)**2*sin((1/32)*pi*y)*cos((1/32)*pi*y) - 0.0001220703125*pi**3*sin((1/32)*pi*y)*sin((1/32)*pi*z)**2*cos((1/32)*pi*y) + 0.0625*pi*sin((1/16)*pi*y)*sin((1/16)*pi*z)*cos((1/16)*pi*x)

def L32_f1(x, y, z):
    return -0.000732421875*pi**3*sin((1/32)*pi*x)*sin((1/32)*pi*y)**2*sin((1/32)*pi*z)**2*cos((1/32)*pi*x) + 0.0001220703125*pi**3*sin((1/32)*pi*x)*sin((1/32)*pi*y)**2*cos((1/32)*pi*x) + 0.0001220703125*pi**3*sin((1/32)*pi*x)*sin((1/32)*pi*z)**2*cos((1/32)*pi*x) + 0.0625*pi*sin((1/16)*pi*x)*sin((1/16)*pi*z)*cos((1/16)*pi*y)

def L32_f2(x, y, z):
    return (1/16)*pi*sin((1/16)*pi*x)*sin((1/16)*pi*y)*cos((1/16)*pi*z)

def L48_u0(x, y, z):
    return 0.0416666666666667*pi*sin(0.0208333333333333*pi*x)**2*sin(0.0208333333333333*pi*y)*sin(0.0208333333333333*pi*z)**2*cos(0.0208333333333333*pi*y)

def L48_u1(x, y, z):
    return -0.0416666666666667*pi*sin(0.0208333333333333*pi*x)*sin(0.0208333333333333*pi*y)**2*sin(0.0208333333333333*pi*z)**2*cos(0.0208333333333333*pi*x)

def L48_u2(x, y, z):
    return 0

def L48_p0(x, y, z):
    return sin(0.0416666666666667*pi*x)*sin(0.0416666666666667*pi*y)*sin(0.0416666666666667*pi*z)

def L48_f0(x, y, z):
    return pi*(0.000217013888888889*pi**2*sin(0.0208333333333333*pi*x)**2*sin(0.0208333333333333*pi*y)*sin(0.0208333333333333*pi*z)**2*cos(0.0208333333333333*pi*y) - 3.61689814814815e-5*pi**2*sin(0.0208333333333333*pi*x)**2*sin(0.0208333333333333*pi*y)*cos(0.0208333333333333*pi*y) - 3.61689814814815e-5*pi**2*sin(0.0208333333333333*pi*y)*sin(0.0208333333333333*pi*z)**2*cos(0.0208333333333333*pi*y) + 0.0416666666666667*sin(0.0416666666666667*pi*y)*sin(0.0416666666666667*pi*z)*cos(0.0416666666666667*pi*x))

def L48_f1(x, y, z):
    return pi*(-0.000217013888888889*pi**2*sin(0.0208333333333333*pi*x)*sin(0.0208333333333333*pi*y)**2*sin(0.0208333333333333*pi*z)**2*cos(0.0208333333333333*pi*x) + 3.61689814814815e-5*pi**2*sin(0.0208333333333333*pi*x)*sin(0.0208333333333333*pi*y)**2*cos(0.0208333333333333*pi*x) + 3.61689814814815e-5*pi**2*sin(0.0208333333333333*pi*x)*sin(0.0208333333333333*pi*z)**2*cos(0.0208333333333333*pi*x) + 0.0416666666666667*sin(0.0416666666666667*pi*x)*sin(0.0416666666666667*pi*z)*cos(0.0416666666666667*pi*y))

def L48_f2(x, y, z):
    return 0.0416666666666667*pi*sin(0.0416666666666667*pi*x)*sin(0.0416666666666667*pi*y)*cos(0.0416666666666667*pi*z)

def L64_u0(x, y, z):
    return (1/32)*pi*sin((1/64)*pi*x)**2*sin((1/64)*pi*y)*sin((1/64)*pi*z)**2*cos((1/64)*pi*y)

def L64_u1(x, y, z):
    return -1/32*pi*sin((1/64)*pi*x)*sin((1/64)*pi*y)**2*sin((1/64)*pi*z)**2*cos((1/64)*pi*x)

def L64_u2(x, y, z):
    return 0

def L64_p0(x, y, z):
    return sin((1/32)*pi*x)*sin((1/32)*pi*y)*sin((1/32)*pi*z)

def L64_f0(x, y, z):
    return 9.1552734375e-5*pi**3*sin((1/64)*pi*x)**2*sin((1/64)*pi*y)*sin((1/64)*pi*z)**2*cos((1/64)*pi*y) - 1.52587890625e-5*pi**3*sin((1/64)*pi*x)**2*sin((1/64)*pi*y)*cos((1/64)*pi*y) - 1.52587890625e-5*pi**3*sin((1/64)*pi*y)*sin((1/64)*pi*z)**2*cos((1/64)*pi*y) + 0.03125*pi*sin((1/32)*pi*y)*sin((1/32)*pi*z)*cos((1/32)*pi*x)

def L64_f1(x, y, z):
    return -9.1552734375e-5*pi**3*sin((1/64)*pi*x)*sin((1/64)*pi*y)**2*sin((1/64)*pi*z)**2*cos((1/64)*pi*x) + 1.52587890625e-5*pi**3*sin((1/64)*pi*x)*sin((1/64)*pi*y)**2*cos((1/64)*pi*x) + 1.52587890625e-5*pi**3*sin((1/64)*pi*x)*sin((1/64)*pi*z)**2*cos((1/64)*pi*x) + 0.03125*pi*sin((1/32)*pi*x)*sin((1/32)*pi*z)*cos((1/32)*pi*y)

def L64_f2(x, y, z):
    return (1/32)*pi*sin((1/32)*pi*x)*sin((1/32)*pi*y)*cos((1/32)*pi*z)

FUNCTIONS = {(16.0, 1.0): {'u': [L16_u0,L16_u1,L16_u2],'p': [L16_p0],'f': [L16_f0,L16_f1,L16_f2]},(24.0, 1.0): {'u': [L24_u0,L24_u1,L24_u2],'p': [L24_p0],'f': [L24_f0,L24_f1,L24_f2]},(32.0, 1.0): {'u': [L32_u0,L32_u1,L32_u2],'p': [L32_p0],'f': [L32_f0,L32_f1,L32_f2]},(48.0, 1.0): {'u': [L48_u0,L48_u1,L48_u2],'p': [L48_p0],'f': [L48_f0,L48_f1,L48_f2]},(64.0, 1.0): {'u': [L64_u0,L64_u1,L64_u2],'p': [L64_p0],'f': [L64_f0,L64_f1,L64_f2]}}


class Manufactured:
    def __init__(self,length,viscosity):
        self.functions=FUNCTIONS[(float(length),float(viscosity))]
    def _eval(self,name,X,Y,Z):
        return [np.broadcast_to(np.asarray(fn(X,Y,Z),dtype=np.float64),X.shape) for fn in self.functions[name]]
    def velocity(self,X,Y,Z):return np.stack(self._eval('u',X,Y,Z),axis=-1)
    def pressure(self,X,Y,Z):return self._eval('p',X,Y,Z)[0]
    def forcing(self,X,Y,Z):return np.stack(self._eval('f',X,Y,Z),axis=-1)

def gauss(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * nodes, 0.5 * weights

def voxel_integrals(field, centres, h, order):
    """Exact-to-quadrature integral of a vector or scalar field over each voxel.

    `centres` has shape (n, 3) in physical units; the return is the integral over
    the voxel, i.e. the mean times h^3.
    """
    offset, weight = gauss(order)
    total = None
    for i, wi in zip(offset, weight):
        for j, wj in zip(offset, weight):
            for k, wk in zip(offset, weight):
                X = centres[:, 0] + i * h
                Y = centres[:, 1] + j * h
                Z = centres[:, 2] + k * h
                value = field(X, Y, Z)
                contribution = (wi * wj * wk) * value
                total = contribution if total is None else total + contribution
    return total * (h ** 3)

def face_integrals(field_component, centroids, normals, h, order):
    """Integral of u.n over each facelet (a voxel face of side h)."""
    offset, weight = gauss(order)
    axis = np.argmax(np.abs(normals), axis=1)
    tangents = np.zeros((centroids.shape[0], 2, 3), dtype=np.float64)
    for a in range(3):
        rows = np.flatnonzero(axis == a)
        others = [t for t in range(3) if t != a]
        tangents[rows, 0, others[0]] = 1.0
        tangents[rows, 1, others[1]] = 1.0
    total = np.zeros(centroids.shape[0], dtype=np.float64)
    for i, wi in zip(offset, weight):
        for j, wj in zip(offset, weight):
            point = centroids + (i * h) * tangents[:, 0, :] + (j * h) * tangents[:, 1, :]
            velocity = field_component(point[:, 0], point[:, 1], point[:, 2])
            total += (wi * wj) * np.sum(velocity * normals, axis=1)
    return total * (h * h)

def lattice_sites(n: int, m: int) -> np.ndarray:
    """Structured nested lattice of m^3 sites inside an n^3 voxel cube."""
    if m > n:
        raise ValueError("more sites per side than voxels per side")
    index = np.rint((np.arange(m) + 0.5) * n / m - 0.5).astype(np.int64)
    index = np.clip(index, 0, n - 1)
    if np.unique(index).size != m:
        raise ValueError("the requested lattice is degenerate at this voxel resolution")
    zz, yy, xx = np.meshgrid(index, index, index, indexing="ij")
    return np.unique((zz * n + yy) * n + xx)

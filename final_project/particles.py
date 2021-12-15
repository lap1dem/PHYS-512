import numpy as np
import numba as nb
from scipy import fft


def _get_kernel(gridsize, soft):
    x = fft.fftfreq(gridsize) * gridsize
    rsqr = np.outer(np.ones(gridsize), x ** 2)
    rsqr = rsqr + rsqr.T
    rsqr[rsqr < soft ** 2] = soft ** 2
    kernel = rsqr ** -0.5
    return kernel


@nb.njit
def _update_rho(x, m, gridsize, rho, idxs, ignore, periodic=False):
    for i in range(x.shape[0]):
        if ignore[i]:
            continue

        ix = int(x[i, 0] + 0.5)
        iy = int(x[i, 1] + 0.5)

        if periodic:
            if ix >= gridsize:
                x[i, 0] = x[i, 0] - gridsize
                ix = ix - gridsize
            elif ix < 0:
                x[i, 0] = x[i, 0] + gridsize
                ix = ix + gridsize
            if iy >= gridsize:
                x[i, 1] = x[i, 1] - gridsize
                iy = iy - gridsize
            elif iy < 0:
                x[i, 1] = x[i, 1] + gridsize
                iy = iy + gridsize
        else:
            if ix >= gridsize or iy >= gridsize or ix < 0 or iy < 0:
                ignore[i] = True
                continue

        rho[ix, iy] = rho[ix, iy] + m[i]
        idxs[i, 0] = ix
        idxs[i, 1] = iy



def _get_grad(pot, periodic=False):
    if periodic:
        vxr = np.roll(pot, 1, axis=0)
        vxl = np.roll(pot, -1, axis=0)
        vyr = np.roll(pot, 1, axis=1)
        vyl = np.roll(pot, -1, axis=1)
    else:
        # Assuming potential = 0 out of borders
        vxr = np.pad(pot, ((1, 0), (0, 0)), mode='constant')[:-1, :]
        vxl = np.pad(pot, ((0, 1), (0, 0)), mode='constant')[1:, :]
        vyr = np.pad(pot, ((0, 0), (1, 0)), mode='constant')[:, :-1]
        vyl = np.pad(pot, ((0, 0), (0, 1)), mode='constant')[:, 1:]

    grady = (vyr - vyl) / 2
    gradx = (vxr - vxl) / 2

    return [gradx, grady]



class Particles:
    def __init__(self, npart: int, soft: float = 1, gridsize: int = 256, periodic=False):
        self.x = np.empty([npart, 2])
        self.ix = np.zeros([npart, 2], dtype=np.int32)
        self.v = np.empty([npart, 2])
        self.m = np.ones(npart)

        self.npart = npart
        self.gridsize = gridsize
        self.soft = soft
        self.periodic = periodic

        # Used to remove particles for non-periodic boundaries
        self.ignore = np.zeros(npart, dtype=bool)

        self.f = np.empty([npart, 2])
        self.grad = [None, None]

        self.kernel = None
        self.kernelft = None

        self.pot = np.empty([self.gridsize, self.gridsize])
        self.rho = np.empty([self.gridsize, self.gridsize])

        self._update_kernel()

    def _inbound_xarray(self):
        self.x[self.x < -0.5] = self.x[self.x < -0.5] + self.gridsize
        self.x[self.x >= self.gridsize - 0.5] = self.x[self.x >= self.gridsize - 0.5] - self.gridsize

    def _update_kernel(self):
        if self.periodic:
            self.kernel = _get_kernel(self.gridsize, self.soft)
        else:
            self.kernel = _get_kernel(self.gridsize * 2, self.soft)

        self.kernelft = fft.rfft2(self.kernel, workers=8)

    def _update_rho(self):
        self.rho[:] = 0.
        _update_rho(self.x, self.m, self.gridsize, self.rho, self.ix, self.ignore, self.periodic)


    def _update_potential(self):
        self._update_rho()

        if self.periodic:
            rhoft = fft.rfft2(self.rho, workers=8)
            self.pot = fft.irfft2(rhoft * self.kernelft, [self.gridsize] * 2, workers=8)
        else:
            rho_padded = np.pad(self.rho,
                                ((0, self.gridsize), (0, self.gridsize)),
                                mode='constant',
                                constant_values=0)
            rhoft = fft.rfft2(rho_padded, workers=8)
            self.pot = fft.irfft2(rhoft * self.kernelft, workers=8)[:self.gridsize, :self.gridsize]

    def _update_forces(self):
        self.grad = _get_grad(self.pot, periodic=self.periodic)
        self.f[:, 0] = - self.grad[0][self.ix[:, 0], self.ix[:, 1]]
        self.f[self.ignore, 0] = 0.
        self.f[:, 1] = - self.grad[1][self.ix[:, 0], self.ix[:, 1]]
        self.f[self.ignore, 1] = 0.

    def leapfrog_shift(self, dt):
        self._update_potential()
        self._update_forces()
        self.v[:] = self.v[:] + self.f * dt

    def take_step(self, dt):
        self.x[:] = self.x[:] + dt * self.v
        self._update_potential()
        self._update_forces()
        self.v[:] = self.v[:] + self.f * dt

    def take_step_rk4(self, dt=1):
        # Preparation
        x0 = self.x
        v0 = self.v
        # Step 1
        self._update_potential()
        self._update_forces()
        v1 = self.v
        f1 = self.f
        # Step 2
        self.x = x0 + v1 * dt / 2
        self.v = v0 + f1 * dt / 2
        self._update_potential()
        self._update_forces()
        v2 = self.v
        f2 = self.f
        # Step 3
        self.x = x0 + v2 * dt / 2
        self.v = v0 + f2 * dt / 2
        self._update_potential()
        self._update_forces()
        v3 = self.v
        f3 = self.f
        # Step 4
        self.x = x0 + v3 * dt
        self.v = v0 + f3 * dt
        self._update_potential()
        self._update_forces()
        v4 = self.v
        f4 = self.f
        # Finally
        self.x = x0 + (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6
        self.v = v0 + (f1 + 2 * f2 + 2 * f3 + f4) * dt / 6

    def setpos_gauss(self, vel=False):
        self.x[:] = np.random.randn(self.npart, 2) * self.gridsize / 8 + self.gridsize / 2
        self._inbound_xarray()
        self.m[:] = 1.
        if not vel:
            self.v[:] = 0.
        else:
            self.v[:] = np.random.randn(self.npart, 2)

    def setpos_uniform(self, vel=False):
        self.x[:] = np.random.uniform(0, self.gridsize, (self.npart, 2))
        self._inbound_xarray()
        self.m[:] = 1.
        if not vel:
            self.v[:] = 0.
        else:
            self.v[:] = np.random.randn(self.npart, 2)
        self._update_potential()
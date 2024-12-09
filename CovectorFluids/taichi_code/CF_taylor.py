import taichi as ti
import matplotlib.pyplot as plt
import numpy as np

@ti.data_oriented
class TaylorGreenVortex:
    def __init__(
        self,
        U=1.0,
        a=0.3,
        distance=0.8,
        res=256,
        max_jacobi_iter=10000
    ):
        self.U = U
        self.a = a
        self.distance = distance
        self.nx = res
        self.ny = res
        self.h = 2 * ti.math.pi / res
        self.max_jacobi_iter = max_jacobi_iter

        self.curl = ti.field(ti.f32, shape=(self.nx+1, self.ny+1))
        self.pressure = ti.field(ti.f32, shape=(self.nx+1, self.ny+1))
        self.pressure_temp = ti.field(ti.f32, shape=(self.nx+1, self.ny+1))
        self.velx = ti.field(ti.f32, shape=(self.nx+1, self.ny))
        self.vely = ti.field(ti.f32, shape=(self.nx, self.ny+1))
        self.vor = ti.field(ti.f32, shape=(self.nx+1, self.ny+1))

        self.pressure.fill(0)
        self.pressure_temp.fill(0)
        self.vor.fill(0)

    @ti.kernel
    def init_curl(self):
        for i, j in self.curl:
            pos = ti.Vector([i, j]) * self.h - ti.Vector([ti.math.pi, ti.math.pi])
            vor_pos1 = ti.Vector([-0.5 * self.distance, 0])
            vor_pos2 = ti.Vector([0.5 * self.distance, 0])
            r_sq1 = (pos - vor_pos1).norm_sqr()
            r_sq2 = (pos - vor_pos2).norm_sqr()
            self.curl[i,j] = self.U/self.a * (2.0 - r_sq1/self.a**2) * ti.exp(0.5*(1.0 - r_sq1/self.a**2)) \
                           + self.U/self.a * (2.0 - r_sq2/self.a**2) * ti.exp(0.5*(1.0 - r_sq2/self.a**2))

    @ti.kernel
    def poisson_jacobi(self):
        for i, j in self.pressure:
            # periodic boundary condition
            # pl = self.pressure[i-1, j] if i > 0 else self.pressure[self.nx, j]
            # pr = self.pressure[i+1, j] if i < self.nx else self.pressure[0, j]
            # pb = self.pressure[i, j-1] if j > 0 else self.pressure[i, self.ny]
            # pt = self.pressure[i, j+1] if j < self.ny else self.pressure[i, 0]

            # Dirichlet boundary condition with zero boundary value
            pl = self.pressure[i-1, j] if i > 0 else 0
            pr = self.pressure[i+1, j] if i < self.nx else 0
            pb = self.pressure[i, j-1] if j > 0 else 0
            pt = self.pressure[i, j+1] if j < self.ny else 0

            self.pressure_temp[i,j] = 0.25 * (pl+pr+pb+pt - self.h**2 * self.curl[i,j])

    def pressure_solve(self):
        for _ in range(self.max_jacobi_iter):
            self.poisson_jacobi()
            self.pressure.copy_from(self.pressure_temp)

    @ti.kernel
    def get_velocity(self):
        for i, j in self.velx:
            self.velx[i,j] = -(self.pressure[i,j+1] - self.pressure[i,j]) / self.h
        for i, j in self.vely:
            self.vely[i,j] = (self.pressure[i+1,j] - self.pressure[i,j]) / self.h

    def generate_field(self):
        self.init_curl()
        self.pressure_solve()
        self.get_velocity()

    @ti.kernel
    def vel_to_vor(self):
        for i, j in ti.ndrange((1, self.nx), (1, self.ny)):
            self.vor[i,j] = (self.vely[i,j] - self.vely[i-1,j] - self.velx[i,j] + self.velx[i,j-1]) / self.h


# ti.init(arch=ti.gpu)
# tgv = TaylorGreenVortex()
# tgv.generate_field()
# tgv.vel_to_vor()

# plt.figure(figsize=(12, 6))
# plt.subplot(1,2,1)
# plt.imshow(tgv.curl.to_numpy(), cmap='jet')
# plt.colorbar()
# plt.title('curl')
# plt.subplot(1,2,2)
# plt.imshow(tgv.vor.to_numpy(), cmap='jet')
# plt.colorbar()
# plt.title('vorticity')
# plt.show()
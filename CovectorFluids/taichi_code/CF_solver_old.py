import taichi as ti

@ti.data_oriented
class CovectorFluidSolver:
    def __init__(self, nx, ny, h, dt):
        self.nx = nx
        self.ny = ny
        self.h = h  # h = dx = dy
        self.h_inv = 1.0 / h
        self.dt = dt
        self.poisson_multiplier = h / dt
        self.max_jacobi_iteration = 1000

        self.velx = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
        self.vely = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
        self.velx_temp = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
        self.vely_temp = ti.field(dtype=ti.f32, shape=(nx, ny + 1))

        self.p = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.p_temp = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.psi = ti.Vector.field(2, dtype=ti.f32, shape=(nx+2, ny+2))
        # self.div = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vor = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))
        self.abs_vor = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))

        self.p.fill(0)
        self.p_temp.fill(0)
        self.vor.fill(0)

    def init_velocity(self, vx, vy):
        if self.velx.shape != vx.shape or self.vely.shape != vy.shape:
            raise ValueError('Invalid input shape')
        self.velx.copy_from(vx)
        self.vely.copy_from(vy)

    def init_vorticity(self, vor):
        if self.vor.shape != vor.shape:
            raise ValueError('Invalid input shape')
        self.vor.copy_from(vor)

    @staticmethod
    @ti.func
    def sample(qfield, i, j):
        ni, nj = qfield.shape
        i = min(max(i, 0), ni - 1)
        j = min(max(j, 0), nj - 1)
        # i = i % ni # periordic boundary condition
        # j = j % nj # periordic boundary condition
        return qfield[i, j]

    @ti.func
    def clampPosition(self, pos):
        x = min(max(pos[0], 0.0*self.h), self.nx*self.h)
        y = min(max(pos[1], 0.0*self.h), self.ny*self.h)
        # periodic boundary condition
        # i = int(pos[0] / self.h); j =  int(pos[1] / self.h)
        # offset_x = pos[0] - i*self.h; offset_y = pos[1] - j*self.h
        # x = (i%self.nx) * self.h + offset_x
        # y = (j%self.ny) * self.h + offset_y
        return ti.Vector([x, y])

    @staticmethod
    @ti.func
    def lerp(vl, vr, frac):
        return vl + frac * (vr - vl)

    @ti.func
    def bilerp(self, v00, v01, v10, v11, fracx, fracy):
        return self.lerp(self.lerp(v00, v01, fracx), self.lerp(v10, v11, fracx), fracy)

    @ti.func
    def getVelocity(self, pos, ux, uy):
        xpos = pos - ti.Vector([0.0, 0.5*self.h])
        i = int(xpos[0] / self.h); j = int(xpos[1] / self.h)
        ux_sample = self.bilerp(self.sample(ux, i, j), self.sample(ux, i+1, j),
                                self.sample(ux, i, j+1), self.sample(ux, i+1, j+1),
                                (xpos[0] / self.h) - i, (xpos[1] / self.h) - j)
        
        ypos = pos - ti.Vector([0.5*self.h, 0.0])
        i = int(ypos[0] / self.h); j = int(ypos[1] / self.h)
        uy_sample = self.bilerp(self.sample(uy, i, j), self.sample(uy, i+1, j),
                                self.sample(uy, i, j+1), self.sample(uy, i+1, j+1),
                                (ypos[0] / self.h) - i, (ypos[1] / self.h) - j)
        
        return ti.Vector([ux_sample, uy_sample])

    @ti.func
    def backtraceRK3(self, pos, ux, uy, dt):
        v1 = self.getVelocity(pos, ux, uy)
        p1 = self.clampPosition(pos - 0.5*dt*v1)
        v2 = self.getVelocity(p1, ux, uy)
        p2 = self.clampPosition(pos - 0.75*dt*v2)
        v3 = self.getVelocity(p2, ux, uy)
        pos_ = pos - dt*((2.0/9.0)*v1 + (1.0/3.0)*v2 + (4.0/9.0)*v3)
        return self.clampPosition(pos_)
    
    @ti.func
    def backtraceRK4(self, pos, ux, uy, dt):
        v1 = self.getVelocity(pos, ux, uy)
        p1 = self.clampPosition(pos - 0.5*dt*v1)
        v2 = self.getVelocity(p1, ux, uy)
        p2 = self.clampPosition(pos - 0.5*dt*v2)
        v3 = self.getVelocity(p2, ux, uy)
        p3 = self.clampPosition(pos - dt*v3)
        v4 = self.getVelocity(p3, ux, uy)
        pos_ = pos - dt*((1.0/6.0)*v1 + (1.0/3.0)*v2 + (1.0/3.0)*v3 + (1.0/6.0)*v4)
        return self.clampPosition(pos_)
    
    @ti.kernel
    def updatePsi(self, ux: ti.template(), uy: ti.template()):
        for i, j in self.psi:
            pos = ti.Vector([i - 0.5, j - 0.5]) * self.h
            self.psi[i, j] = self.backtraceRK4(pos, ux, uy, self.dt)

    @ti.kernel
    def advection_CF_SL(self, dt: ti.template()):
        for i,j in self.velx:
            v_back = self.getVelocity(
                self.backtraceRK4(ti.Vector([i*self.h, (j+0.5)*self.h]), self.velx, self.vely, dt),
                self.velx, self.vely
            )
            dPsidx = self.h_inv * (self.psi[i+1, j+1] - self.psi[i, j+1])
            self.velx_temp[i,j] = dPsidx[0]*v_back[0] + dPsidx[1]*v_back[1]
        for i,j in self.vely:
            v_back = self.getVelocity(
                self.backtraceRK4(ti.Vector([(i+0.5)*self.h, j*self.h]), self.velx, self.vely, dt),
                self.velx, self.vely
            )
            dPsidy = self.h_inv * (self.psi[i+1, j+1] - self.psi[i+1, j])
            self.vely_temp[i,j] = dPsidy[0]*v_back[0] + dPsidy[1]*v_back[1]

    @ti.kernel
    def advection_SL(self):
        for i,j in self.velx_temp:
            coord = self.backtraceRK4(ti.Vector([i*self.h, (j+0.5)*self.h]), self.velx, self.vely, self.dt)
            self.velx_temp[i,j] = self.getVelocity(coord, self.velx, self.vely)[0]
        for i,j in self.vely_temp:
            coord = self.backtraceRK4(ti.Vector([(i+0.5)*self.h, j*self.h]), self.velx, self.vely, self.dt)
            self.vely_temp[i,j] = self.getVelocity(coord, self.velx, self.vely)[1]

    @ti.kernel
    def poisson_jacobi(self, qf: ti.template(), new_qf: ti.template(), vx: ti.template(), vy: ti.template()):
        for i, j in ti.ndrange(self.nx, self.ny):
            # Dirichlet boundary condition with zero boundary value
            ql = qf[i-1, j] if i > 0 else 0
            qr = qf[i+1, j] if i < self.nx-1 else 0
            qb = qf[i, j-1] if j > 0 else 0
            qt = qf[i, j+1] if j < self.ny-1 else 0
            # periodic boundary condition
            # ql = qf[i-1, j] if i > 0 else qf[self.nx-1, j]
            # qr = qf[i+1, j] if i < self.nx-1 else qf[0, j]
            # qb = qf[i, j-1] if j > 0 else qf[i, self.ny-1]
            # qt = qf[i, j+1] if j < self.ny-1 else qf[i, 0]
            
            div = vx[i+1, j] - vx[i, j] + vy[i, j+1] - vy[i, j]
            new_qf[i, j] = 0.25 * (ql + qr + qb + qt - self.poisson_multiplier*div)

    def projection(self):
        for _ in range(self.max_jacobi_iteration):
            self.poisson_jacobi(self.p, self.p_temp, self.velx_temp, self.vely_temp)
            self.p.copy_from(self.p_temp)

    @ti.kernel
    def apply_pressure(self):
        for i, j in ti.ndrange(self.nx+1, self.ny):
            # Dirichlet boundary condition with zero boundary value
            pl = self.p[i-1, j] if i > 0 else 0
            pr = self.p[i, j] if i < self.nx else 0
            # periodic boundary condition
            # pl = self.p[i-1, j] if i > 0 else self.p[self.nx-1, j]
            # pr = self.p[i, j] if i < self.nx else self.p[0, j]
            self.velx_temp[i, j] -= (self.dt / self.h) * (pr - pl)
        for i, j in ti.ndrange(self.nx, self.ny+1):
            # Dirichlet boundary condition with zero boundary value
            pb = self.p[i, j-1] if j > 0 else 0
            pt = self.p[i, j] if j < self.ny else 0
            # periodic boundary condition
            # pb = self.p[i, j-1] if j > 0 else self.p[i, self.ny-1]
            # pt = self.p[i, j] if j < self.ny else self.p[i, 0]
            self.vely_temp[i, j] -= (self.dt / self.h) * (pt - pb)

    @ti.kernel
    def vel_to_vor(self):
        for i, j in ti.ndrange((1, self.nx), (1, self.ny)):
            self.vor[i,j] = (self.vely[i,j] - self.vely[i-1,j] - self.velx[i,j] + self.velx[i,j-1]) / self.h
            self.abs_vor[i,j] = abs(self.vor[i,j])

    def advance(self):
        self.updatePsi(self.velx, self.vely)
        # self.advection_SL()
        self.advection_CF_SL(self.dt)
        self.projection()
        self.apply_pressure()
        self.velx.copy_from(self.velx_temp)
        self.vely.copy_from(self.vely_temp)
        self.vel_to_vor()

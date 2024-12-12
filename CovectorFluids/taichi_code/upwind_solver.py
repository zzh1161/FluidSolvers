import taichi as ti

@ti.data_oriented
class UpwindSolver:
    def __init__(
        self,
        nx, ny, h, dt,
        max_jacobi_iteration=1000,
    ):
        self.nx = nx
        self.ny = ny
        self.h = h
        self.dt = dt
        self.max_jacobi_iteration = max_jacobi_iteration

        self.velx = ti.field(dtype=ti.f32, shape=(self.nx+1, self.ny))
        self.velx_temp = ti.field(dtype=ti.f32, shape=(self.nx+1, self.ny))
        self.vely = ti.field(dtype=ti.f32, shape=(self.nx, self.ny+1))
        self.vely_temp = ti.field(dtype=ti.f32, shape=(self.nx, self.ny+1))
        self.p = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.p_temp = ti.field(dtype=ti.f32, shape=(nx, ny))

        self.div = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vor = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))
        self.abs_vor = ti.field(dtype=ti.f32, shape=(nx+1, ny+1))

        self.velx.fill(0)
        self.vely.fill(0)
        self.p.fill(0)
        self.div.fill(0)
        self.vor.fill(0)
        self.abs_vor.fill(0)

    def init_velocity(self, vx, vy):
        if self.velx.shape != vx.shape or self.vely.shape != vy.shape:
            raise ValueError('Invalid input shape')
        self.velx.copy_from(vx)
        self.vely.copy_from(vy)

    @ti.func
    def sample(self, field, i, j):
        ni, nj = field.shape
        res = 0.0
        if i >= 0 and i < ni and j >= 0 and j < nj:
            res = field[i, j]
        return res

    @ti.kernel
    def advection_upwind_1st(
        self,
        dt: ti.template(),
        vx: ti.template(), vy: ti.template(),
        new_vx: ti.template(), new_vy: ti.template()
    ):
        for i, j in vx:
            flux_x = (self.sample(vx, i, j) - self.sample(vx, i-1, j)) / self.h if self.sample(vx, i, j) > 0 \
                     else (self.sample(vx, i+1, j) - self.sample(vx, i, j)) / self.h
            flux_y = (self.sample(vx, i, j) - self.sample(vx, i, j-1)) / self.h if self.sample(vy, i, j) > 0 \
                     else (self.sample(vx, i, j+1) - self.sample(vx, i, j)) / self.h
            new_vx[i, j] = self.sample(vx, i, j) - dt * (self.sample(vx, i, j)*flux_x + self.sample(vy, i, j)*flux_y)
        for i, j in vy:
            flux_x = (self.sample(vy, i, j) - self.sample(vy, i-1, j)) / self.h if self.sample(vx, i, j) > 0 \
                     else (self.sample(vy, i+1, j) - self.sample(vy, i, j)) / self.h
            flux_y = (self.sample(vy, i, j) - self.sample(vy, i, j-1)) / self.h if self.sample(vy, i, j) > 0 \
                     else (self.sample(vy, i, j+1) - self.sample(vy, i, j)) / self.h
            new_vy[i, j] = self.sample(vy, i, j) - dt * (self.sample(vx, i, j)*flux_x + self.sample(vy, i, j)*flux_y)

    @ti.kernel
    def poisson_jacobi(
        self,
        dt: ti.template(),
        pf: ti.template(), new_pf: ti.template(),
        vx: ti.template(), vy: ti.template()
    ):
        '''
        Single Jacobi iteration for solving a Poisson equation
        pf: current field
        new_pf: updated field
        vx, vy: flow velocity field, whose divergence constitutes the rhs
        '''
        for i, j in pf:
            pl = pf[i-1, j] if i > 0 else 0
            pr = pf[i+1, j] if i < self.nx-1 else 0
            pb = pf[i, j-1] if j > 0 else 0
            pt = pf[i, j+1] if j < self.ny-1 else 0
            
            div = vx[i+1, j] - vx[i, j] + vy[i, j+1] - vy[i, j]
            new_pf[i, j] = 0.25 * (pl + pr + pb + pt - (self.h/dt) * div)

    def pressure_projection(
        self,
        dt: ti.template(),
        pf: ti.template(), new_pf: ti.template(),
        vx: ti.template(), vy: ti.template()
    ):
        for _ in range(self.max_jacobi_iteration):
            self.poisson_jacobi(dt, pf, new_pf, vx, vy)
            pf.copy_from(new_pf)

    @ti.kernel
    def apply_pressure(
        self,
        dt: ti.template(),
        pf: ti.template(), 
        vx: ti.template(), vy: ti.template()
    ):
        for i, j in vx:
            pl = pf[i-1, j] if i > 0 else 0
            pr = pf[i, j] if i < self.nx else 0
            vx[i, j] -= (dt / self.h) * (pr - pl)
        for i, j in vy:
            pb = pf[i, j-1] if j > 0 else 0
            pt = pf[i, j] if j < self.ny else 0
            vy[i, j] -= (dt / self.h) * (pt - pb)

    @ti.kernel
    def vel_to_vor(self):
        for i, j in ti.ndrange((1, self.nx), (1, self.ny)):
            self.vor[i,j] = (self.vely[i,j] - self.vely[i-1,j] -
                             self.velx[i,j] + self.velx[i,j-1]) / self.h
            self.abs_vor[i,j] = abs(self.vor[i,j])

    def advance(self):
        self.advection_upwind_1st(self.dt, self.velx, self.vely, self.velx_temp, self.vely_temp)
        self.velx.copy_from(self.velx_temp)
        self.vely.copy_from(self.vely_temp)
        self.pressure_projection(self.dt, self.p, self.p_temp, self.velx, self.vely)
        self.apply_pressure(self.dt, self.p, self.velx, self.vely)
        self.vel_to_vor()
import taichi as ti
from enum import Enum

class BoundaryCond(Enum):
    DIRICHLET = 0
    PERIODIC = 1

class Scheme(Enum):
    CF_SL_1ST = 0
    CF_SL_2ND = 1
    CF_BFECC = 2
    CF_MCM = 3

@ti.data_oriented
class CovectorFluidSolver:
    def __init__(
        self,
        nx, ny, h, dt,
        max_jacobi_iteration=1000,
        bc=BoundaryCond.DIRICHLET,
        scheme=Scheme.CF_SL_2ND
    ):
        self.nx = nx
        self.ny = ny
        self.h = h # h = dx = dy
        self.h_inv = 1.0 / h
        self.dt = dt
        self.max_jacobi_iteration = max_jacobi_iteration
        self.bc = bc
        self.scheme = scheme

        self.velx = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
        self.vely = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
        self.velx_temp = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
        self.vely_temp = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
        # For BFECC
        self.vx0 = ti.field(dtype=ti.f32, shape=(self.nx + 1, self.ny))
        self.vy0 = ti.field(dtype=ti.f32, shape=(self.nx, self.ny + 1))
        self.ex = ti.field(dtype=ti.f32, shape=(self.nx + 1, self.ny))
        self.ey = ti.field(dtype=ti.f32, shape=(self.nx, self.ny + 1))

        # Forward map and backward map
        self.fwd_map = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.fwd_map_temp = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.bwd_map = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.bwd_map_temp = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        # For BiMocq
        self.vx1 = ti.field(dtype=ti.f32, shape=(self.nx + 1, self.ny))
        self.vy1 = ti.field(dtype=ti.f32, shape=(self.nx, self.ny + 1))
        self.vx0_ = ti.field(dtype=ti.f32, shape=(self.nx + 1, self.ny))
        self.vy0_ = ti.field(dtype=ti.f32, shape=(self.nx, self.ny + 1))
        self.errx = ti.field(dtype=ti.f32, shape=(self.nx + 1, self.ny))
        self.erry = ti.field(dtype=ti.f32, shape=(self.nx, self.ny + 1))

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

        self.reinit_mcm()

    def init_velocity(self, vx, vy):
        if self.velx.shape != vx.shape or self.vely.shape != vy.shape:
            raise ValueError('Invalid input shape')
        self.velx.copy_from(vx)
        self.vely.copy_from(vy)

    def init_vorticity(self, vor):
        if self.vor.shape != vor.shape:
            raise ValueError('Invalid input shape')
        self.vor.copy_from(vor)

    @ti.kernel
    def maxVelocity(self) -> ti.f32:
        max_vel = ti.f32(0.0)
        for i, j in self.velx:
            max_vel = ti.max(max_vel, ti.abs(self.velx[i,j]))
        for i, j in self.vely:
            max_vel = ti.max(max_vel, ti.abs(self.vely[i,j]))
        return max_vel

    @staticmethod
    @ti.kernel
    def field_minus(a: ti.template(), b: ti.template(), res: ti.template()):
        for I in ti.grouped(a):
            res[I] = a[I] - b[I]
    
    @staticmethod
    @ti.kernel
    def field_plus(a: ti.template(), b: ti.template(), res: ti.template()):
        for I in ti.grouped(a):
            res[I] = a[I] + b[I]
        
    @staticmethod
    @ti.kernel
    def field_multiply(a: ti.template(), res: ti.template()):
        for I in ti.grouped(res):
            res[I] = a * res[I]

    @ti.func
    def sample(self, qfield, i, j):
        ni, nj = qfield.shape
        i_ = min(max(i, 0), ni - 1)
        j_ = min(max(j, 0), nj - 1)
        if ti.static(self.bc == BoundaryCond.PERIODIC):
            i_ = i % ni
            j_ = j % nj
        return qfield[i_, j_]
    
    @ti.func
    def clampPosition(self, pos):
        x = min(max(pos[0], 0.0*self.h), self.nx*self.h)
        y = min(max(pos[1], 0.0*self.h), self.ny*self.h)
        if ti.static(self.bc == BoundaryCond.PERIODIC):
            i = int(pos[0] / self.h); j =  int(pos[1] / self.h)
            offset_x = pos[0] - i*self.h; offset_y = pos[1] - j*self.h
            x = (i%self.nx) * self.h + offset_x
            y = (j%self.ny) * self.h + offset_y
        return ti.Vector([x, y])
    
    @staticmethod
    @ti.func
    def lerp(vl, vr, frac):
        return vl + frac * (vr - vl)

    @ti.func
    def bilerp(self, v00, v01, v10, v11, fracx, fracy):
        return self.lerp(self.lerp(v00, v01, fracx), self.lerp(v10, v11, fracx), fracy)
    
    @ti.func
    def sampleField(self, offset_pos, field):
        i = int(offset_pos[0] / self.h); j = int(offset_pos[1] / self.h)
        return self.bilerp(self.sample(field, i, j), self.sample(field, i+1, j),
                           self.sample(field, i, j+1), self.sample(field, i+1, j+1),
                           (offset_pos[0] / self.h) - i, (offset_pos[1] / self.h) - j)

    @ti.func
    def getVelocity(self, pos, ux, uy):
        xpos = pos - ti.Vector([0.0, 0.5*self.h])
        ux_sample = self.sampleField(xpos, ux)
        
        ypos = pos - ti.Vector([0.5*self.h, 0.0])
        uy_sample = self.sampleField(ypos, uy)
        
        return ti.Vector([ux_sample, uy_sample])
    
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
    def advection_CF_SL(
        self,
        dt: ti.template(),
        ux: ti.template(), uy: ti.template(),
        vx: ti.template(), vy: ti.template(),
        new_ux: ti.template(), new_uy: ti.template()
    ):
        '''
        Covector advection with semi-Lagrangian scheme: A(u; v, dt)
        ux, uy: field to advect
        vx, vy: flow velocity field
        new_ux, new_uy: advected field
        NOTE: new_u gotta be different from u and v!
        '''
        # if ux.shape != new_ux.shape or uy.shape != new_uy.shape:
        #     raise ValueError('Invalid input shape')
        for i, j in ux:
            backpos_face = self.backtraceRK4(ti.Vector([i*self.h, (j+0.5)*self.h]), vx, vy, dt)
            backpos_vol_r = self.backtraceRK4(ti.Vector([(i+0.5)*self.h, (j+0.5)*self.h]), vx, vy, dt)
            backpos_vol_l = self.backtraceRK4(ti.Vector([(i-0.5)*self.h, (j+0.5)*self.h]), vx, vy, dt)
            dPsidx = (backpos_vol_r - backpos_vol_l) / self.h
            u_back = self.getVelocity(backpos_face, ux, uy)
            new_ux[i,j] = dPsidx[0]*u_back[0] + dPsidx[1]*u_back[1]
        for i, j in uy:
            backpos_face = self.backtraceRK4(ti.Vector([(i+0.5)*self.h, j*self.h]), vx, vy, dt)
            backpos_vol_t = self.backtraceRK4(ti.Vector([(i+0.5)*self.h, (j+0.5)*self.h]), vx, vy, dt)
            backpos_vol_b = self.backtraceRK4(ti.Vector([(i+0.5)*self.h, (j-0.5)*self.h]), vx, vy, dt)
            dPsidy = (backpos_vol_t - backpos_vol_b) / self.h
            u_back = self.getVelocity(backpos_face, ux, uy)
            new_uy[i,j] = dPsidy[0]*u_back[0] + dPsidy[1]*u_back[1]

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
        vx, vy: flow velocity field, whose divergence is the rhs
        '''
        for i, j in pf:
            pl = pf[i-1, j] if i > 0 else 0
            pr = pf[i+1, j] if i < self.nx-1 else 0
            pb = pf[i, j-1] if j > 0 else 0
            pt = pf[i, j+1] if j < self.ny-1 else 0
            if ti.static(self.bc == BoundaryCond.PERIODIC):
                pl = pf[i-1, j] if i > 0 else pf[-1, j]
                pr = pf[i+1, j] if i < self.nx-1 else pf[0, j]
                pb = pf[i, j-1] if j > 0 else pf[i, -1]
                pt = pf[i, j+1] if j < self.ny-1 else pf[i, 0]
            
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
            if ti.static(self.bc == BoundaryCond.PERIODIC):
                pl = pf[i-1, j] if i > 0 else pf[-1, j]
                pr = pf[i, j] if i < self.nx else pf[0, j]
            vx[i, j] -= (dt / self.h) * (pr - pl)
        for i, j in vy:
            pb = pf[i, j-1] if j > 0 else 0
            pt = pf[i, j] if j < self.ny else 0
            if ti.static(self.bc == BoundaryCond.PERIODIC):
                pb = pf[i, j-1] if j > 0 else pf[i, -1]
                pt = pf[i, j] if j < self.ny else pf[i, 0]
            vy[i, j] -= (dt / self.h) * (pt - pb)
    
    @ti.kernel
    def vel_to_vor(self):
        for i, j in ti.ndrange((1, self.nx), (1, self.ny)):
            self.vor[i,j] = (self.vely[i,j] - self.vely[i-1,j] -
                             self.velx[i,j] + self.velx[i,j-1]) / self.h
            self.abs_vor[i,j] = abs(self.vor[i,j])

    #####################################################
    # Methods for CF+MCM
    #####################################################
    @ti.kernel
    def reinit_mcm(self):
        for i, j in self.fwd_map:
            self.fwd_map[i,j] = ti.Vector([(i+0.5)*self.h, (j+0.5)*self.h])
            self.bwd_map[i,j] = ti.Vector([(i+0.5)*self.h, (j+0.5)*self.h])
        for i, j in self.velx:
            self.vx0[i,j] = self.velx[i,j]
        for i, j in self.vely:
            self.vy0[i,j] = self.vely[i,j]

    @ti.kernel
    def update_bwdmap(
        self,
        dt: ti.template(),
        psi: ti.template(),
        vx: ti.template(), vy: ti.template(),
        new_psi: ti.template()
    ):
        for i, j in psi:
            backpos = self.backtraceRK4(ti.Vector([(i+0.5)*self.h, (j+0.5)*self.h]), vx, vy, dt)
            new_psi[i,j] = self.sampleField(backpos - ti.Vector([0.5*self.h, 0.5*self.h]), psi)

    @ti.kernel
    def pullbackVelcoity(
        self,
        psi: ti.template(),
        ux0: ti.template(), uy0: ti.template(),
        ux1: ti.template(), uy1: ti.template()
    ):
        '''
        u1 = dPsi^T (u0 o Psi)
        '''
        for i, j in ux1:
            pos_r = self.sample(psi, i, j)
            pos_l = self.sample(psi, i-1, j)
            dPsidx = (pos_r - pos_l) / self.h
            u0 = self.getVelocity(0.5*(pos_r + pos_l), ux0, uy0)
            ux1[i,j] = dPsidx[0]*u0[0] + dPsidx[1]*u0[1]
        for i, j in uy1:
            pos_t = self.sample(psi, i, j)
            pos_b = self.sample(psi, i, j-1)
            dPsidy = (pos_t - pos_b) / self.h
            u0 = self.getVelocity(0.5*(pos_t + pos_b), ux0, uy0)
            uy1[i,j] = dPsidy[0]*u0[0] + dPsidy[1]*u0[1]

    @ti.kernel
    def update_fwdmap(
        self,
        dt: ti.template(),
        phi: ti.template(),
        vx: ti.template(), vy: ti.template(),
        new_phi: ti.template()
    ):
        for i, j in phi:
            fwdpos = phi[i,j]
            new_phi[i,j] = self.backtraceRK4(fwdpos, vx, vy, -dt)

    @ti.kernel
    def distortion(self) -> ti.f32:
        d = ti.f32(0.0)
        for i, j in ti.ndrange((1, self.nx), (1, self.ny)):
            pos = ti.Vector([(i+0.5)*self.h, (j+0.5)*self.h])
            backpos = self.bwd_map[i,j]
            repos1 = self.sampleField(backpos - ti.Vector([0.5*self.h, 0.5*self.h]), self.fwd_map)
            dist1 = (repos1 - pos).norm()
            fwdpos = self.fwd_map[i,j]
            repos2 = self.sampleField(fwdpos - ti.Vector([0.5*self.h, 0.5*self.h]), self.bwd_map)
            dist2 = (repos2 - pos).norm()
            dist = max(dist1, dist2)
            if dist > d:
                d = dist
        return d

    def advance(self):
        if ti.static(self.scheme == Scheme.CF_SL_1ST):
            self.advection_CF_SL(self.dt,
                                 self.velx, self.vely,
                                 self.velx, self.vely,
                                 self.velx_temp, self.vely_temp)
            self.velx.copy_from(self.velx_temp)
            self.vely.copy_from(self.vely_temp)
            self.pressure_projection(self.dt, self.p, self.p_temp, self.velx, self.vely)
            self.apply_pressure(self.dt, self.p, self.velx, self.vely)
        elif ti.static(self.scheme == Scheme.CF_SL_2ND):
            self.advection_CF_SL(0.5*self.dt,
                                 self.velx, self.vely,
                                 self.velx, self.vely,
                                 self.velx_temp, self.vely_temp)
            self.pressure_projection(0.5*self.dt, self.p, self.p_temp, self.velx_temp, self.vely_temp)
            self.apply_pressure(0.5*self.dt, self.p, self.velx_temp, self.vely_temp)
            self.advection_CF_SL(self.dt,
                                 self.velx, self.vely,
                                 self.velx_temp, self.vely_temp,
                                 self.vx0, self.vy0)
            self.velx.copy_from(self.vx0)
            self.vely.copy_from(self.vy0)
            self.pressure_projection(self.dt, self.p, self.p_temp, self.velx, self.vely)
            self.apply_pressure(self.dt, self.p, self.velx, self.vely)
        elif ti.static(self.scheme == Scheme.CF_BFECC):
            self.advection_CF_SL(self.dt,
                                 self.velx, self.vely,
                                 self.velx, self.vely,
                                 self.velx_temp, self.vely_temp)
            self.advection_CF_SL(-self.dt,
                                 self.velx_temp, self.vely_temp,
                                 self.velx, self.vely,
                                 self.vx0, self.vy0)
            self.field_minus(self.vx0, self.velx, self.ex)
            self.field_minus(self.vy0, self.vely, self.ey)
            self.field_multiply(0.5, self.ex)
            self.field_multiply(0.5, self.ey)
            self.advection_CF_SL(self.dt,
                                 self.ex, self.ey,
                                 self.velx, self.vely,
                                 self.vx0, self.vy0)
            self.field_minus(self.velx_temp, self.vx0, self.velx)
            self.field_minus(self.vely_temp, self.vy0, self.vely)
            self.pressure_projection(self.dt, self.p, self.p_temp, self.velx, self.vely)
            self.apply_pressure(self.dt, self.p, self.velx, self.vely)
        elif ti.static(self.scheme == Scheme.CF_MCM):
            # Step 1: estimate velocity
            self.velx_temp.copy_from(self.velx)
            self.vely_temp.copy_from(self.vely)
            # Step 2: advect the inverse flow map
            self.update_bwdmap(self.dt, self.bwd_map, self.velx, self.vely, self.bwd_map_temp)
            self.bwd_map.copy_from(self.bwd_map_temp)
            # Step 3: pullback velocity
            self.pullbackVelcoity(self.bwd_map, self.vx0, self.vy0, self.vx1, self.vy1)
            # Step 4: march flow map
            self.update_fwdmap(self.dt, self.fwd_map, self.velx_temp, self.vely_temp, self.fwd_map_temp)
            self.fwd_map.copy_from(self.fwd_map_temp)
            # Step 5: back-and-forth transport
            self.pullbackVelcoity(self.fwd_map, self.vx1, self.vy1, self.vx0_, self.vy0_)
            # Step 6: roundtrip error
            self.field_minus(self.vx0_, self.velx, self.ex)
            self.field_minus(self.vy0_, self.vely, self.ey)
            self.field_multiply(0.5, self.ex)
            self.field_multiply(0.5, self.ey)
            # Step 7: error correction
            self.pullbackVelcoity(self.bwd_map, self.ex, self.ey, self.errx, self.erry)
            self.field_minus(self.vx1, self.errx, self.velx)
            self.field_minus(self.vy1, self.erry, self.vely)
            # Step 8: pressure projection
            self.pressure_projection(self.dt, self.p, self.p_temp, self.velx, self.vely)
            self.apply_pressure(self.dt, self.p, self.velx, self.vely)
            # Step 9: reinitialization judgement
            dist = self.distortion()
            maxvel = self.maxVelocity()
            if dist / (maxvel*self.dt) > 0.1:
                self.reinit_mcm()

        self.vel_to_vor()
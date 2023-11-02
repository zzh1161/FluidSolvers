import taichi as ti
import numpy as np
from time import time
import taichi.math as tm
import Simulation.simulation_setting

from utils.macros import *
import random

@ti.data_oriented
class Grid3DSimulator:
    def __init__(
            self,
            n_grid=[64,64,64],
            range_grid=[2,2,2],
            use_particle=True
        ):
        self.dim = 3
        self.n_grid_x,self.n_grid_y,self.n_grid_z=n_grid[0],n_grid[1],n_grid[2]
        self.grid_x,self.grid_y,self.grid_z=range_grid[0],range_grid[1],range_grid[2]
        self.dx,self.dy,self.dz=self.grid_x/self.n_grid_x,self.grid_y/self.n_grid_y,self.grid_z/self.n_grid_z
        
        self.use_particle=use_particle
        self.rng_state=0
    
    def set_particles(self,particle_number,particle_x,particle_v,particle_valid):
            self.particle_x=particle_x
            self.particle_v=particle_v
            self.particle_valid=particle_valid
            self.particle_number=particle_number

    def update_particle(self,dt,v):
        self.update_position_RK4(dt,v)
        #self.update_velocity(v)
        #self.update_position(dt)

    @ti.kernel
    def update_velocity(self,v:ti.template()):
        for i in range(self.particle_number):
            self.particle_v[i]=self.from_grid_to_particle(self.particle_x[i],v)

    @ti.kernel
    def update_position(self,dt:DTYPE_TI):
        for i in range(self.particle_number):
            if(self.particle_valid[i]==1):
                self.particle_x[i]= self.particle_x[i]+self.particle_v[i]*dt             
            else:
                self.particle_v[i]=(0.,0.,0.)

            self.regular_position(i)
     
    @ti.kernel
    def update_position_RK4(self,dt:DTYPE_TI,v:ti.template()):
        for i in range(self.particle_number):
            if(self.particle_valid[i]==1):
                k1=self.from_grid_to_particle(self.particle_x[i],v)
                k2=self.from_grid_to_particle(self.particle_x[i]+k1*dt/2,v)
                k3=self.from_grid_to_particle(self.particle_x[i]+k2*dt/2,v)
                k4=self.from_grid_to_particle(self.particle_x[i]+k3*dt,v) 
                self.particle_v[i]=(k1 + 2 * k2 + 2 * k3 + k4)/6
                self.particle_x[i]= self.particle_x[i]+self.particle_v[i]*dt
            else:
                self.particle_v[i]=(0.,0.,0.)

            self.particle_x[i]=self.regular_position(self.particle_x[i])

    @ti.func
    def from_grid_to_particle(self,pos,v_field):
        """
            https://blog.csdn.net/weixin_42795611/article/details/111566400
        """
        pos=self.regular_position(pos)
        idx=pos/vec3d_ti(self.dx,self.dy,self.dz)
        idxi=int(pos//vec3d_ti(self.dx,self.dy,self.dz))

        next_idxi=int((idxi+1)%vec3d_ti(self.n_grid_x,self.n_grid_y,self.n_grid_z))
        w=idx-idxi
        vx_00=v_field[idxi[0],idxi[1],idxi[2]]*(1-w[0])+v_field[next_idxi[0],idxi[1],idxi[2]]*w[0]
        vx_01=v_field[idxi[0],idxi[1],next_idxi[2]]*(1-w[0])+v_field[next_idxi[0],idxi[1],next_idxi[2]]*w[0]
        vx_10=v_field[idxi[0],next_idxi[1],idxi[2]]*(1-w[0])+v_field[next_idxi[0],next_idxi[1],idxi[2]]*w[0]
        vx_11=v_field[idxi[0],next_idxi[1],next_idxi[2]]*(1-w[0])+v_field[next_idxi[0],next_idxi[1],next_idxi[2]]*w[0]

        vy0=vx_00*(1-w[1])+vx_10*w[1]
        vy1=vx_01*(1-w[1])+vx_11*w[1]

        v=vy0*(1-w[2])+vy1*w[2]

        return v

    @ti.func
    def regular_position(self,pos):
        res=pos
        if(pos[0]>self.grid_x):
            res[0]-=self.grid_x
        if(pos[1]>self.grid_y):
            res[1]-=self.grid_y
        if(pos[2]>self.grid_z):
            res[2]-=self.grid_z
        if(pos[0]<0):
            res[0]+=self.grid_x
        if(pos[1]<0):
            res[1]+=self.grid_y
        if(pos[2]<0):
            res[2]+=self.grid_z 
        return res

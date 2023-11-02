import taichi as ti
import taichi.math as tm
import random

import schrodinger_setting

from utils.macros import *

@ti.data_oriented
class SchrodingerExample:
    def __init__(
        self,
        n_grid=[64,64,64],
        range_grid=[2,2,2]
    ):
        self.dim=3
        self.n_grid=vec3i_ti(n_grid[0],n_grid[1],n_grid[2])
        self.range_grid=vec3d_ti(range_grid[0],range_grid[1],range_grid[2])
        self.dxyz=self.range_grid/self.n_grid

        self.grid_wave1r=ti.field(DTYPE_TI, shape=(self.n_grid[0], self.n_grid[1], self.n_grid[2]))
        self.grid_wave1i=ti.field(DTYPE_TI, shape=(self.n_grid[0], self.n_grid[1], self.n_grid[2]))
        self.grid_wave2r=ti.field(DTYPE_TI, shape=(self.n_grid[0], self.n_grid[1], self.n_grid[2]))
        self.grid_wave2i=ti.field(DTYPE_TI, shape=(self.n_grid[0], self.n_grid[1], self.n_grid[2]))
    
    def generate_particle(self):
        pass

    @ti.func
    def length_complex(self,z1):
        return tm.sqrt(z1[0]*z1[0]+z1[1]*z1[1])
    
    @ti.func
    def length(self,z1,z2):
        return tm.sqrt(z1[0]*z1[0]+z1[1]*z1[1]+z2[0]*z2[0]+z2[1]*z2[1])
    
    @ti.func
    def normalize(self,z1,z2):
        len_z=self.length(z1,z2)
        return z1[0]/len_z,z1[1]/len_z,z2[0]/len_z,z2[1]/len_z
                                           
    @ti.func
    def complex_multiple(self,z1,z2):
        return (z1[0]*z2[0]-z1[1]*z2[1]),(z1[1]*z2[0]+z1[0]*z2[1])
    
    @ti.func
    def complex_add(self,z1,z2):
        return (z1[0]+z2[0],z1[1]+z2[1])

@ti.data_oriented
class LeapFrog(SchrodingerExample):
    def __init__(
        self,
        n_grid=[128,64,64],
        range_grid=[10,5,5],
        bg_v=[-0.2, 0, 0],
        bg_omega=0,
        cylinder_normal1=[-1,0,0],
        cylinder_normal2=[-1,0,0],
        cylinder_center1=[5,2.5,2.5],
        cylinder_center2=[5,2.5,2.5],
        cylinder_radius1=20*5/64,
        cylinder_radius2=12*5/64,
        cylinder_thickness1=5*5/64,
        cylinder_thickness2=5*5/64
    ):
        super().__init__(n_grid,range_grid)
        self.bg_v=vec3d_ti(bg_v)
        self.bg_omega=bg_omega
        self.cylinder_normal1=vec3d_ti(cylinder_normal1)
        self.cylinder_normal2=vec3d_ti(cylinder_normal2)
        self.cylinder_center1=vec3d_ti(cylinder_center1)
        self.cylinder_center2=vec3d_ti(cylinder_center2)
        self.cylinder_radius1=cylinder_radius1
        self.cylinder_radius2=cylinder_radius2
        self.cylinder_thickness1=cylinder_thickness1
        self.cylinder_thickness2=cylinder_thickness2

    def init(self):
        self.init_wave()
        self.background_velocity(self.bg_v/h_plank,self.bg_omega)
        self.cylinder_wave(self.cylinder_radius1,self.cylinder_thickness1,self.cylinder_normal1,self.cylinder_center1)
        self.cylinder_wave(self.cylinder_radius2,self.cylinder_thickness2,self.cylinder_normal2,self.cylinder_center2)

    @ti.kernel
    def init_wave(self):
        for i,j,k in self.grid_wave1r:
            self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]=1,0
            self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]=0.01,0
            self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k],self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]=self.normalize(
                (self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]),(self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k])
            )
   
    @ti.kernel
    def background_velocity(self,bg_v:vec3d_ti,bg_omega:DTYPE_TI):
        for i,j,k in self.grid_wave1r:
            phase_v=bg_v*vec3d_ti(i,j,k)*self.range_grid/self.n_grid
            phase=phase_v[0]+phase_v[1]+phase_v[2]-bg_omega
            amp1=self.length_complex((self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]))
            amp2=self.length_complex((self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]))
            self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]=amp1*tm.cos(phase),amp1*tm.sin(phase)
            self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]=amp2*tm.cos(phase),amp2*tm.sin(phase)

    @ti.kernel
    def cylinder_wave(
        self,
        cylinder_radius: DTYPE_TI,
        cylinder_thickness: DTYPE_TI,
        cylinder_normal: ti.types.vector(3, DTYPE_TI),
        cylinder_center: ti.types.vector(3, DTYPE_TI)
    ):
        for i,j,k in self.grid_wave1r:
            dist_vec=vec3d_ti(i,j,k)*self.dxyz-cylinder_center
            z=tm.dot(cylinder_normal,dist_vec)
            dist=tm.dot(dist_vec,dist_vec)
            if(dist-z*z<cylinder_radius*cylinder_radius):
                alpha=ti.cast(0,DTYPE_TI)
                if(z>0 and z<cylinder_thickness/2):
                    alpha=-tm.pi*(2*z/cylinder_thickness-1)
                elif(z<=0 and z>=-cylinder_thickness/2):
                    alpha=-tm.pi*(2*z/cylinder_thickness+1)
                self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]=self.complex_multiple(
                    (self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]),
                    (tm.cos(alpha),tm.sin(alpha))
                )
                #print(self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k])
    
    def generate_particle(
            self,
            particle_number=65536,
            box_size=(1.0*5/64, 30.0*5/64, 30.0*5/64),
            box_center=(5, 2.5, 2.5)
        ):
        self.particle_number=particle_number
        self.particle_x=ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_number)
        self.particle_v=ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.particle_number)
        self.particle_valid=ti.field(dtype=DTYPE_TI, shape=self.particle_number)
        self.init_valid()
        self.generate_particle_positioin(vec3d_ti(box_size),vec3d_ti(box_center))
        
    @ti.kernel
    def init_valid(self):
        for i in range(self.particle_number):
            self.particle_valid[i]=1

    def generate_particle_positioin(self,box_size,box_center):
        for i in range(self.particle_number):
            pos = (vec3d_ti(random.random(),random.random(),random.random())*2-1)
            pos *= box_size
            pos +=box_center
            self.particle_x[i] = pos


@ti.data_oriented
class Nozzle(SchrodingerExample):
    def __init__(
        self,
        n_grid=[128,64,64],
        range_grid=[20,16,16],
        bg_omega=0,
        nozzle_center = [4, 8, 8],
        nozzle_dir = [1, 0, 0],
        nozzle_velocity = [1, 0, 0],
        nozzle_radius = 3,
        nozzle_length = 3
    ):
        super().__init__(n_grid,range_grid)
        
        self.bg_omega=bg_omega
        self.nozzle_center=vec3d_ti(nozzle_center)
        self.nozzle_dir=vec3d_ti(nozzle_dir)
        self.nozzle_velocity=vec3d_ti(nozzle_velocity)
        self.nozzle_radius=nozzle_radius
        self.nozzle_length=nozzle_length
        # handle the data
        self.nozzle_dir,self.nozzle_right,self.nozzle_up=self.handle_data()
        # set the nozzle field
        self.nozzle=ti.field(DTYPE_TI, shape=(self.n_grid[0], self.n_grid[1], self.n_grid[2]))

    def length_vec3d(self,v):
        return math.sqrt(v[0]**2+v[1]**2+v[2]**2)
    
    def normalize_vec3d(self,v):
        return v/self.length_vec3d(v)
    
    def cross_vec3d(self,v1,v2):
        return vec3d_ti(
            v1[1]*v2[2]-v2[1]*v1[2],
            -v1[0]*v2[2]+v2[0]*v1[2],
            v1[0]*v2[1]-v1[1]*v2[0]
        )
    
    def handle_data(self):
        nozzle_dir= self.normalize_vec3d(self.nozzle_dir)
        if(self.length_vec3d(nozzle_dir-vec3d_ti(0,1,0))<1e-14):
            nozzle_dir[0] += 0.01
        nozzle_right = self.cross_vec3d(nozzle_dir, vec3d_ti(0,1,0))
        nozzle_right= self.normalize_vec3d(nozzle_right)
        nozzle_up = self.cross_vec3d(nozzle_dir, nozzle_right)
        nozzle_up=self.normalize_vec3d(nozzle_up)

        return nozzle_dir,nozzle_right,nozzle_up

    def init(self):
        self.init_wave()
        self.init_nozzle(self.nozzle_radius,self.nozzle_length,self.nozzle_dir,self.nozzle_center)
        self.background_velocity(self.nozzle_velocity/2/h_plank,self.bg_omega)

    @ti.kernel
    def init_wave(self):
        for i,j,k in self.grid_wave1r:
            self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]=1,0
            self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]=0.01,0
            self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k],self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]=self.normalize(
                (self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]),(self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k])
            )
   
    @ti.kernel
    def background_velocity(self,bg_v:vec3d_ti,bg_omega:DTYPE_TI):
        for i,j,k in self.nozzle:
            if(not self.nozzle[i,j,k] == 0):
                phase_v=self.nozzle_velocity*vec3d_ti(i,j,k)*self.dxyz
                phase=phase_v[0]+phase_v[1]+phase_v[2]-bg_omega
                amp1=self.length_complex((self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]))
                amp2=self.length_complex((self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]))
                self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]=amp1*tm.cos(phase),amp1*tm.sin(phase)
                self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]=amp2*tm.cos(phase),amp2*tm.sin(phase)

    @ti.kernel
    def init_nozzle(
        self,
        nozzle_radius: DTYPE_TI,
        nozzle_length: DTYPE_TI,
        nozzle_dir: ti.types.vector(3, DTYPE_TI),
        nozzle_center: ti.types.vector(3, DTYPE_TI)
    ):
        for i,j,k in self.nozzle:
            dist_vec=vec3d_ti(i,j,k)*self.dxyz-nozzle_center
            dir=tm.dot(nozzle_dir,dist_vec)
            m=dist_vec-dir*nozzle_dir
            if(abs(dir)<nozzle_length/2):
                if(tm.length(m)<nozzle_radius):
                    self.nozzle[i,j,k]=1
                else:
                    self.nozzle[i,j,k]=0
            else:
                self.nozzle[i,j,k]=0

    def generate_particle(
            self,
            particle_number=65536
        ):
        self.particle_number=particle_number
        self.particle_x=ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_number)
        self.particle_v=ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.particle_number)
        self.particle_valid=ti.field(dtype=DTYPE_TI, shape=self.particle_number)
        self.init_valid()
        self.generate_particle_positioin()
        
    @ti.kernel
    def init_valid(self):
        for i in range(self.particle_number):
            self.particle_valid[i]=1

    def generate_particle_positioin(self):
        for i in range(self.particle_number):
            t=random.random()*tm.pi*4
            disrtube,disrtube2,disrtube3=random.random()*tm.pi*0.02,random.random()*tm.pi*0.02,random.random()*tm.pi*0.02
            r = (self.nozzle_up * tm.cos(t) + disrtube) + (self.nozzle_right * tm.sin(t) + disrtube2)
            r *= self.nozzle_radius * 0.9
            r += self.nozzle_center
            r += (disrtube3 - 0.03) * self.nozzle_dir * 100
            self.particle_x[i]=r



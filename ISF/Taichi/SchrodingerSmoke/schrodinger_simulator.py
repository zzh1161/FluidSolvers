import taichi as ti
import numpy as np
from time import time
import taichi.math as tm
import schrodinger_setting

from utils.macros import *
import Simulation.fft_gpu as fft_gpu
from Simulation.grid_simulator import Grid3DSimulator

@ti.data_oriented
class SchrodingerSimulator(Grid3DSimulator):
    def __init__(
            self,
            h_plank,
            n_grid=[64,64,64],
            range_grid=[2,2,2],
            total_time=1e-2,
            dt=1e-4,
            archetech_select="cpu", # or gpu,
            use_particle=True
        ):
        # setting value
        ## geometry setting value
        super().__init__(n_grid,range_grid,use_particle)

        ## physical setting value
        self.total_time=total_time
        self.dt=dt
        self.h_plank=h_plank
        # archetect value
        self.archetech_select=archetech_select
    
    def build(
        self
        # TODO
        ):
        self.setup_fields() 
        
    def setup_fields(self):
        # main field for calculation
        self.grid_wave1r=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.grid_wave1i=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.grid_wave2r=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.grid_wave2i=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.grid_pressure = ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.grid_divergence = ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.eta_x=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.eta_y=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.eta_z=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        
        # multiplier which only compute once
        self.schrodinger_integration_multiplier1=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.schrodinger_integration_multiplier2=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.pressure_project_multiplier=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))

        # temp variable
        self.tem_grid_wave1r=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.tem_grid_wave1i=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.tem_grid_wave2r=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))
        self.tem_grid_wave2i=ti.field(DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))

        # to convert to others
        self.velocity=ti.Vector.field(3, dtype=DTYPE_TI, shape=(self.n_grid_x, self.n_grid_y, self.n_grid_z))


    ############################################################
    ################# taichi kernels ###########################
    ############################################################

    def initialize(
            self,
            field1r=None,
            field1i=None,
            field2r=None,
            field2i=None
        ):
        if(field1r is not None and field1i is not None):
            self.grid_wave1r=field1r
            self.grid_wave1i=field1i
        else:
            print("You do not set wave filed psi 1")

        if(field2r is not None and field2i is not None):
            self.grid_wave2r=field2r
            self.grid_wave2i=field2i
        else:
            print("You do not set wave filed psi 2")

        self.initialize_wave()
        
    def initialize_wave(self):
        self.normalize_wave(self.grid_wave1r,self.grid_wave1i,self.grid_wave2r,self.grid_wave2i)
        self.pressure_project_init()


    @ti.kernel
    def precomputing(self):
        for i,j,k in self.schrodinger_integration_multiplier1:
            self.schrodinger_integration_multiplier1[i,j,k],self.schrodinger_integration_multiplier2[i,j,k]=self.compute_schrodinger_integration_multipier(i,j,k,self.dt)
            self.pressure_project_multiplier[i,j,k]=self.compute_pressure_project_multipier(i,j,k)

    def update(self):
        self.schrodinger_integration()
        self.normalize_wave(self.tem_grid_wave1r,self.tem_grid_wave1i,self.tem_grid_wave2r,self.tem_grid_wave2i)
        self.pressure_project()
        self.velocity_update()
        if(self.use_particle):
            self.update_particle(self.dt,self.velocity)
        
    def schrodinger_integration(self):

        # first, do fft
        grid_wave1r_np = self.grid_wave1r.to_numpy()
        grid_wave1i_np = self.grid_wave1i.to_numpy()
        grid_wave2r_np = self.grid_wave2r.to_numpy()
        grid_wave2i_np = self.grid_wave2i.to_numpy()
        
        if(self.archetech_select=="gpu"):
            wave_freq1r,wave_freq1i=fft_gpu.fft3_complex_gpu(grid_wave1r_np,grid_wave1i_np,True)
            wave_freq2r,wave_freq2i=fft_gpu.fft3_complex_gpu(grid_wave2r_np,grid_wave2i_np,True)
        else:
            wave_freq1r,wave_freq1i=fft_gpu.fft3_complex_cpu(grid_wave1r_np,grid_wave1i_np,True)
            wave_freq2r,wave_freq2i=fft_gpu.fft3_complex_cpu(grid_wave2r_np,grid_wave2i_np,True)

        # then multiply
        self.schrodinger_integration_multiply(wave_freq1r,wave_freq1i,wave_freq2r,wave_freq2i)

        # final, do ifft
        if(self.archetech_select=="gpu"):
            grid_wave1r_np,grid_wave1i_np=fft_gpu.ifft3_complex_gpu(wave_freq1r,wave_freq1i,True)
            grid_wave2r_np,grid_wave2i_np=fft_gpu.ifft3_complex_gpu(wave_freq2r,wave_freq2i,True)
        else:
            grid_wave1r_np,grid_wave1i_np=fft_gpu.ifft3_complex_cpu(wave_freq1r,wave_freq1i,True)
            grid_wave2r_np,grid_wave2i_np=fft_gpu.ifft3_complex_cpu(wave_freq2r,wave_freq2i,True)
        
        self.tem_grid_wave1r.from_numpy(grid_wave1r_np)
        self.tem_grid_wave1i.from_numpy(grid_wave1i_np)
        self.tem_grid_wave2r.from_numpy(grid_wave2r_np)
        self.tem_grid_wave2i.from_numpy(grid_wave2i_np)

    @ti.kernel
    def normalize_wave(self,grid_wave1r:ti.template(),grid_wave1i:ti.template(),grid_wave2r:ti.template(),grid_wave2i:ti.template()):
        for i,j,k in grid_wave1i:
            grid_wave1r[i,j,k],grid_wave1i[i,j,k],grid_wave2r[i,j,k],grid_wave2i[i,j,k]=self.normalize(
                (grid_wave1r[i,j,k],grid_wave1i[i,j,k]),
                (grid_wave2r[i,j,k],grid_wave2i[i,j,k])
            )

    def pressure_project(self):
        self.velocity_one_form_scaled(self.tem_grid_wave1r,self.tem_grid_wave1i,self.tem_grid_wave2r,self.tem_grid_wave2i)
        self.divergence()

        print("divergence",self.grid_divergence[5,5,5])
        
        grid_divergence_np=self.grid_divergence.to_numpy()
        if(self.archetech_select=="gpu"):
            grid_divergence_freq1,grid_divergence_freq2=fft_gpu.fft3_gpu(grid_divergence_np)
        else:
            grid_divergence_freq1,grid_divergence_freq2=fft_gpu.fft3_cpu(grid_divergence_np)
        self.pressure_project_multiply(grid_divergence_freq1,grid_divergence_freq2)
        if(self.archetech_select=="gpu"):
            grid_divergence_np=fft_gpu.ifft3_gpu(grid_divergence_freq1,grid_divergence_freq2)
        else:
            grid_divergence_np=fft_gpu.ifft3_cpu(grid_divergence_freq1,grid_divergence_freq2)

        self.pressure_project_final(grid_divergence_np,self.tem_grid_wave1r,self.tem_grid_wave1i,self.tem_grid_wave2r,self.tem_grid_wave2i)
        
        self.velocity_one_form_scaled(self.grid_wave1r,self.grid_wave1i,self.grid_wave2r,self.grid_wave2i)
        self.divergence()
        print("divergence",self.grid_divergence[5,5,5])

    def pressure_project_init(self):
        self.velocity_one_form_scaled(self.grid_wave1r,self.grid_wave1i,self.grid_wave2r,self.grid_wave2i)
        self.divergence()
        grid_divergence_np=self.grid_divergence.to_numpy()
        
        if(self.archetech_select=="gpu"):
            grid_divergence_freq1,grid_divergence_freq2=fft_gpu.fft3_gpu(grid_divergence_np)
        else:
            grid_divergence_freq1,grid_divergence_freq2=fft_gpu.fft3_cpu(grid_divergence_np)
        self.pressure_project_multiply(grid_divergence_freq1,grid_divergence_freq2)
        if(self.archetech_select=="gpu"):
            grid_divergence_np=fft_gpu.ifft3_gpu(grid_divergence_freq1,grid_divergence_freq2)
        else:
            grid_divergence_np=fft_gpu.ifft3_cpu(grid_divergence_freq1,grid_divergence_freq2)

        self.pressure_project_final(grid_divergence_np,self.grid_wave1r,self.grid_wave1i,self.grid_wave2r,self.grid_wave2i)

    #Note:ndarray are passed by reference
    @ti.kernel
    def schrodinger_integration_multiply(
        self,
        wave_freq1r:ti.types.ndarray(),
        wave_freq1i:ti.types.ndarray(),
        wave_freq2r:ti.types.ndarray(),
        wave_freq2i:ti.types.ndarray()
    ):
        # do multiply
        for i,j,k in self.grid_wave1r:
            wave_freq1r[i,j,k],wave_freq1i[i,j,k]=self.complex_multiple(
                    (self.schrodinger_integration_multiplier1[i,j,k],self.schrodinger_integration_multiplier2[i,j,k]),
                    (wave_freq1r[i,j,k],wave_freq1i[i,j,k])
                )
            wave_freq2r[i,j,k],wave_freq2i[i,j,k]=self.complex_multiple(
                    (self.schrodinger_integration_multiplier1[i,j,k],self.schrodinger_integration_multiplier2[i,j,k]),
                    (wave_freq2r[i,j,k],wave_freq2i[i,j,k])
                )

    @ti.kernel
    def velocity_one_form_scaled(self,grid_wave1r:ti.template(),grid_wave1i:ti.template(),grid_wave2r:ti.template(),grid_wave2i:ti.template()):
        for i,j,k in self.eta_x:
            self.compute_velocity_one_form_scaled(i,j,k,grid_wave1r,grid_wave1i,grid_wave2r,grid_wave2i)

    @ti.kernel
    def velocity_update(self):
        for i,j,k in self.eta_x:
            self.compute_velocity(i,j,k)
        

    @ti.kernel
    def divergence(self):
        for i,j,k in self.grid_divergence:
            self.grid_divergence[i,j,k]=self.compute_divergence(i,j,k)

    #Note:ndarray are passed by reference
    @ti.kernel
    def pressure_project_multiply(self,divergence_freq1:ti.types.ndarray(),divergence_freq2:ti.types.ndarray()):
        for i,j,k in self.grid_wave1r:
            divergence_freq1[i,j,k]=divergence_freq1[i,j,k]*self.pressure_project_multiplier[i,j,k]
            divergence_freq2[i,j,k]=divergence_freq2[i,j,k]*self.pressure_project_multiplier[i,j,k]

    @ti.kernel
    def pressure_project_final(self,divergence:ti.types.ndarray(),input_wave1r:ti.template(),input_wave1i:ti.template(),input_wave2r:ti.template(),input_wave2i:ti.template()):
        for i,j,k in self.grid_wave1r:
            self.grid_wave1r[i,j,k],self.grid_wave1i[i,j,k]=self.complex_multiple(
                (tm.cos(-divergence[i,j,k]),tm.sin(-divergence[i,j,k])),
                (input_wave1r[i,j,k],input_wave1i[i,j,k])
            )
            self.grid_wave2r[i,j,k],self.grid_wave2i[i,j,k]=self.complex_multiple(
                (tm.cos(-divergence[i,j,k]),tm.sin(-divergence[i,j,k])),
                (input_wave2r[i,j,k],input_wave2i[i,j,k])
            )

    ################################################
    ############# taichi functions #################
    ################################################
    @ti.func
    def complex_multiple(self,z1,z2):
        return (z1[0]*z2[0]-z1[1]*z2[1]),(z1[1]*z2[0]+z1[0]*z2[1])
    
    @ti.func
    def complex_add(self,z1,z2):
        return (z1[0]+z2[0],z1[1]+z2[1])
    
    @ti.func
    def length(self,z1,z2):
        return tm.sqrt(z1[0]*z1[0]+z1[1]*z1[1]+z2[0]*z2[0]+z2[1]*z2[1])
    
    def length_n(self,z1,z2):
        return tm.sqrt(z1[0]*z1[0]+z1[1]*z1[1]+z2[0]*z2[0]+z2[1]*z2[1])    
    
    @ti.func
    def normalize(self,z1,z2):
        len_z=self.length(z1,z2)
        return z1[0]/len_z,z1[1]/len_z,z2[0]/len_z,z2[1]/len_z

    @ti.func
    def compute_velocity_one_form_scaled(self, i, j, k,grid_wave1r,grid_wave1i,grid_wave2r,grid_wave2i):
        nbs_i,nbs_j,nbs_k=(i+1)%self.n_grid_x, (j+1)%self.n_grid_y, (k+1)%self.n_grid_z
        z1=(grid_wave1r[i,j,k],-grid_wave1i[i,j,k])
        z2=(grid_wave2r[i,j,k],-grid_wave2i[i,j,k])
        nx1=(grid_wave1r[nbs_i,j,k],grid_wave1i[nbs_i,j,k])
        nx2=(grid_wave2r[nbs_i,j,k],grid_wave2i[nbs_i,j,k])
        ny1=(grid_wave1r[i,nbs_j,k],grid_wave1i[i,nbs_j,k])
        ny2=(grid_wave2r[i,nbs_j,k],grid_wave2i[i,nbs_j,k])
        nz1=(grid_wave1r[i,j,nbs_k],grid_wave1i[i,j,nbs_k])
        nz2=(grid_wave2r[i,j,nbs_k],grid_wave2i[i,j,nbs_k])
        etaz_x=self.complex_add(self.complex_multiple(z1,nx1),self.complex_multiple(z2,nx2))
        etaz_y=self.complex_add(self.complex_multiple(z1,ny1),self.complex_multiple(z2,ny2))
        etaz_z=self.complex_add(self.complex_multiple(z1,nz1),self.complex_multiple(z2,nz2))
        self.eta_x[i,j,k]=tm.atan2(etaz_x[1],etaz_x[0])
        self.eta_y[i,j,k]=tm.atan2(etaz_y[1],etaz_y[0])
        self.eta_z[i,j,k]=tm.atan2(etaz_z[1],etaz_z[0])

    @ti.func
    def compute_velocity(self, i, j, k):
        self.compute_velocity_one_form_scaled(i,j,k,self.grid_wave1r,self.grid_wave1i,self.grid_wave2r,self.grid_wave2i)
        self.velocity[i,j,k] = vec3d_ti(
            self.eta_x[i,j,k],
            self.eta_y[i,j,k],
            self.eta_z[i,j,k]
        )*self.h_plank/vec3d_ti(self.dx,self.dy,self.dz)

    @ti.func
    def compute_divergence(self, i, j, k):
        s_yz,s_xz,s_xy=self.dy*self.dz,self.dx*self.dz,self.dx*self.dy
        l_x,l_y,l_z=self.dx,self.dy,self.dz
        V=l_x*l_y*l_z
        last_i,last_j,last_k=(i-1+self.n_grid_x)%self.n_grid_x, (j-1+self.n_grid_y)%self.n_grid_y, (k-1+self.n_grid_z)%self.n_grid_z
        diverge=ti.cast(0,DTYPE_TI)
        diverge+=s_yz/l_x*(self.eta_x[i,j,k]-self.eta_x[last_i,j,k])
        diverge+=s_xz/l_y*(self.eta_y[i,j,k]-self.eta_y[i,last_j,k])
        diverge+=s_xy/l_z*(self.eta_z[i,j,k]-self.eta_z[i,j,last_k])
        diverge=diverge/V
        return diverge
    
    @ti.func
    def compute_schrodinger_integration_multipier(self,i,j,k,dt):
        i,j,k=i-self.n_grid_x/2,j-self.n_grid_y/2,k-self.n_grid_z/2
        lambda_schrodinger=-4*tm.pi*tm.pi*(i*i/self.grid_x/self.grid_x+j*j/self.grid_y/self.grid_y+k*k/self.grid_z/self.grid_z)
        z=lambda_schrodinger*dt*self.h_plank/2
        return tm.cos(z),tm.sin(z)

    @ti.func
    def compute_pressure_project_multipier(self,i,j,k):
        lambda_schrodinger=ti.cast(0,DTYPE_TI)
        lambda_schrodinger+=-4/self.dx/self.dx*tm.sin(tm.pi*i/self.n_grid_x)*tm.sin(tm.pi*i/self.n_grid_x)
        lambda_schrodinger+=-4/self.dy/self.dy*tm.sin(tm.pi*j/self.n_grid_y)*tm.sin(tm.pi*j/self.n_grid_y)
        lambda_schrodinger+=-4/self.dz/self.dz*tm.sin(tm.pi*k/self.n_grid_z)*tm.sin(tm.pi*k/self.n_grid_z)
        lambda_schrodinger_inverse=1/lambda_schrodinger
        #print("lambda_schrodinger",lambda_schrodinger)
        if(abs(lambda_schrodinger)==0):
            lambda_schrodinger_inverse=0
        #print("lambda_schrodinger_inverse",lambda_schrodinger_inverse)
        return lambda_schrodinger_inverse



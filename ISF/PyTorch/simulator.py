import torch
import torch.nn as nn
import numpy as np

class schroediger_simulator_2d(nn.Module):
    def __init__(
            self,
            n_grid = [512,512],
            hbar = 0.1,
            dt = 0.03,
            range_x = [-torch.pi,torch.pi],
            range_y = [-torch.pi,torch.pi]
            ):
        super(schroediger_simulator_2d, self).__init__()
        xs = range_x[0]; xe = range_x[1]
        ys = range_y[0]; ye = range_y[1]
        self.hbar = hbar
        self.dt = dt
        self.grid_x = n_grid[0]
        self.grid_y = n_grid[1]
        self.lx = xe - xs
        self.ly = ye - ys
        self.dx = self.lx / n_grid[0]
        self.dy = self.ly / n_grid[1]

        self.ix = torch.tensor(data=range(self.grid_x), dtype=torch.float32)
        self.iy = torch.tensor(data=range(self.grid_y), dtype=torch.float32)
        self.iix, self.iiy = torch.meshgrid(self.ix, self.iy)

        x = self.ix * self.dx + xs
        y = self.iy * self.dy + ys
        self.mesh_x, self.mesh_y = torch.meshgrid(x, y)
        
        a = self.ix + self.grid_x/2
        b = self.iy + self.grid_y/2
        a = complex(0,1) * (a%self.grid_x - self.grid_x/2)
        b = complex(0,1) * (b%self.grid_y - self.grid_y/2)
        self.kx, self.ky = torch.meshgrid(a, b)
        k2 = self.kx*self.kx + self.ky*self.ky
        k2[0,0] = -1
        self.k2 = k2

        kd0 = torch.zeros(self.grid_x, self.grid_y)
        kd1 = torch.ones(self.grid_x, self.grid_y)
        self.kd = torch.where(self.k2.real>-(self.grid_x**2+self.grid_y**2)/18., kd1, kd0)
        self.kdx = self.kx * self.kd
        self.kdy = self.ky * self.kd
        self.kd2 = self.k2 * self.kd

        self.SchroedingerMask = torch.exp(complex(0,0.5)*self.hbar*self.dt*self.k2)

    def Schroedinger(self, psi1, psi2):
        return torch.fft.ifft2(torch.fft.fft2(psi1)*self.SchroedingerMask),\
               torch.fft.ifft2(torch.fft.fft2(psi2)*self.SchroedingerMask)
    
    def Normalization(self, psi1, psi2):
        psi_norm = torch.sqrt(torch.abs(psi1)**2 + torch.abs(psi2)**2)
        return psi1/psi_norm, psi2/psi_norm
    
    def Projection(self, psi1, psi2):
        div = torch.conj(torch.fft.ifft2(torch.fft.fft2(psi1)*self.k2))*complex(0,1)*psi1 +\
              torch.conj(torch.fft.ifft2(torch.fft.fft2(psi2)*self.k2))*complex(0,1)*psi2
        div = torch.exp(complex(0,-1)*torch.fft.ifft2(torch.fft.fft2(div.real)/self.k2))
        return psi1*div, psi2*div
    
    def forward(self, Delta_t, psi1, psi2):
        sub_steps = round(Delta_t / self.dt)
        for _ in range(sub_steps):
            psi1, psi2 = self.Schroedinger(psi1, psi2)
            psi1, psi2 = self.Normalization(psi1, psi2)
            psi1, psi2 = self.Projection(psi1, psi2)
        return psi1, psi2
    
    @staticmethod
    def psi_to_vel(psi1, psi2, kx, ky, hbar):
        psi = torch.fft.fft2(psi1)
        psix = torch.fft.ifft2(psi*kx)
        psiy = torch.fft.ifft2(psi*ky)
        ux = torch.conj(psix)*complex(0,1)*psi1
        uy = torch.conj(psiy)*complex(0,1)*psi1
        psi = torch.fft.fft2(psi2)
        psix = torch.fft.ifft2(psi*kx)
        psiy = torch.fft.ifft2(psi*ky)
        ux = ux + torch.conj(psix)*complex(0,1)*psi2
        uy = uy + torch.conj(psiy)*complex(0,1)*psi2
        return ux.real*hbar, uy.real*hbar
    
    def psi_to_velocity(self, psi1, psi2, hbar):
        psi = torch.fft.fft2(psi1)
        psix = torch.fft.ifft2(psi*self.kx)
        psiy = torch.fft.ifft2(psi*self.ky)
        ux = torch.conj(psix)*complex(0,1)*psi1
        uy = torch.conj(psiy)*complex(0,1)*psi1
        psi = torch.fft.fft2(psi2)
        psix = torch.fft.ifft2(psi*self.kx)
        psiy = torch.fft.ifft2(psi*self.ky)
        ux = ux + torch.conj(psix)*complex(0,1)*psi2
        uy = uy + torch.conj(psiy)*complex(0,1)*psi2
        return ux.real*hbar, uy.real*hbar
import torch
import math
from simulator import schroediger_simulator_2d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
os.chdir(os.path.dirname(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def tube_vortex(mesh_x, mesh_y):
    c1 = [-torch.pi/2, -torch.pi/2]
    c2 = [0, 0]
    rc = [torch.pi/3, torch.pi/6]
    psi1 = torch.ones(mesh_x.shape, dtype = torch.complex64)
    psi2 = 0.01 * torch.ones(mesh_x.shape, dtype = torch.complex64)
    psi1 = psi1.to(device)
    psi2 = psi2.to(device)
    for i in range(2):
        rx = (mesh_x-c1[i])/rc[i]
        ry = (mesh_y-c2[i])/rc[i]
        r2 = rx**2+ry**2
        De = torch.exp(-(r2/9)**4)
        psi1 = psi1 * torch.complex(2*rx*De/(r2+1), (r2+1-2*De)/(r2+1))
    return psi1, psi2

def to_np(x):
    return x.detach().cpu().numpy()

def vel_to_vor(ux, uy, kx, ky):
    return torch.fft.ifft2(kx*torch.fft.fft2(uy)-ky*torch.fft.fft2(ux)).real

def vor_to_vel(wz, kx, ky, k2):
    fwz = torch.fft.fft2(wz)
    return -torch.fft.ifft2(fwz*ky/k2).real, torch.fft.ifft2(fwz*kx/k2).real

if __name__ == '__main__':
    dt = 0.03
    Delta_t = 0.3
    grid_x = 512
    grid_y = 512
    grid_t = 101
    hbar = 0.1

    fig = plt.figure()

    solver = schroediger_simulator_2d()
    solver.to(device)

    x_np = to_np(torch.squeeze(solver.mesh_x))
    y_np = to_np(torch.squeeze(solver.mesh_y))

    psi1, psi2 = tube_vortex(solver.mesh_x, solver.mesh_y)
    psi1, psi2 = solver.Normalization(psi1, psi2)
    psi1 = torch.fft.ifft2(torch.fft.fft2(psi1)*solver.kd)
    psi2 = torch.fft.ifft2(torch.fft.fft2(psi2)*solver.kd)

    for ii in range(10):
        psi1, psi2 = solver.Projection(psi1, psi2)
    
    for i_step in range(grid_t):
        ux,uy = schroediger_simulator_2d.psi_to_vel(psi1,psi2, solver.kx, solver.ky, solver.hbar)
        wz = vel_to_vor(ux,uy,solver.kx,solver.ky)
        w_np = to_np(torch.squeeze(wz))
        vmax = 2
        vmin = -2
        levels = np.linspace(vmin, vmax, 20)
        cmap = mpl.cm.get_cmap('jet', 20)
        cs = plt.contourf(x_np,y_np,w_np,cmap=cmap,vmin=vmin,vmax=vmax,levels=levels)
        plt.pause(0.1)
        plt.savefig('results/'+str(i_step)+'.jpg')
        [psi1,psi2] = solver(Delta_t,psi1,psi2)
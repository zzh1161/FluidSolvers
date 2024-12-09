import taichi as ti
import time
import matplotlib.pyplot as plt
from CF_taylor import TaylorGreenVortex
from CF_solver import CovectorFluidSolver
from CF_solver import Scheme

ti.init(arch=ti.gpu)

nx = 256
ny = 256
L = 2*ti.math.pi
h = L/nx
dt = 0.025
total_frame = 6000

taylorVor = TaylorGreenVortex(res=nx)
solver_1 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_SL_1ST)
solver_2 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_SL_2ND)
solver_b = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_BFECC)
solver_m = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_MCM)

taylorVor.generate_field()
solver_1.init_velocity(taylorVor.velx, taylorVor.vely)
solver_2.init_velocity(taylorVor.velx, taylorVor.vely)
solver_b.init_velocity(taylorVor.velx, taylorVor.vely)
solver_m.init_velocity(taylorVor.velx, taylorVor.vely)

gui = ti.GUI('Taylor Green Vortex', (nx+1, ny+1))
cur_frame = 0
timer = 0
while cur_frame < total_frame:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
    solver_2.advance()
    gui.set_image(solver_2.abs_vor)
    delta_t = time.time() - timer
    timer = time.time()
    gui.text(content=f'fps: {(1.0/delta_t):.1f}', pos=(0,0.98), color=0xffaa77)
    gui.show()
    cur_frame += 1

# cur_frame = 0
# plt.figure(figsize=(18, 6))
# # plt.figure(figsize=(6,6))
# while cur_frame < total_frame:
#     solver_1.advance()
#     solver_2.advance()
#     solver_b.advance()
#     # solver_m.advance()
#     plt.clf()
#     plt.subplot(1,3,1)
#     plt.imshow(solver_1.abs_vor.to_numpy().transpose(), cmap='jet')
#     plt.title('CF_SL_1st')
#     plt.colorbar()
#     plt.subplot(1,3,2)
#     plt.imshow(solver_2.abs_vor.to_numpy().transpose(), cmap='jet')
#     plt.title('CF_SL_2nd')
#     plt.colorbar()
#     plt.subplot(1,3,3)
#     plt.imshow(solver_b.abs_vor.to_numpy().transpose(), cmap='jet')
#     plt.title('CF_BFECC')
#     plt.colorbar()
#     plt.savefig(f'./output/{cur_frame}.png')
#     # plt.imshow(solver_m.abs_vor.to_numpy().transpose(), cmap='jet')
#     # plt.title('CF_MCM')
#     # plt.colorbar()
#     # plt.savefig(f'./output/{cur_frame}.png')
#     cur_frame += 1
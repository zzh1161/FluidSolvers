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
total_frame = 500

taylorVor = TaylorGreenVortex(res=nx)
solver_1 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_SL_1ST)
solver_2 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_SL_2ND)
solver_b = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_BFECC)
solver_m = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_MCM_2ND)

taylorVor.generate_field()
solver_1.init_velocity(taylorVor.velx, taylorVor.vely)
solver_2.init_velocity(taylorVor.velx, taylorVor.vely)
solver_b.init_velocity(taylorVor.velx, taylorVor.vely)
solver_m.init_velocity(taylorVor.velx, taylorVor.vely)

# gui = ti.GUI('Taylor Green Vortex', (nx+1, ny+1))
# cur_frame = 0
# timer = 0
# while cur_frame < total_frame:
#     for e in gui.get_events(ti.GUI.PRESS):
#         if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
#             exit()
#     solver_m.advance()
#     gui.set_image(solver_m.abs_vor)
#     delta_t = time.time() - timer
#     timer = time.time()
#     gui.text(content=f'fps: {(1.0/delta_t):.1f}', pos=(0,0.98), color=0xffaa77)
#     gui.show()
#     cur_frame += 1

cur_frame = 0
vmin = 0.0
vmax = 10.0
while cur_frame < total_frame:
    solver_1.advance()
    solver_2.advance()
    solver_b.advance()
    solver_m.advance()

    plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # subplot 1
    im1 = axs[0, 0].imshow(solver_1.abs_vor.to_numpy().transpose(), cmap='jet', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('1st order Semi-Lagrangian')
    axs[0, 0].axis('off')
    # subplot 2
    axs[0, 1].imshow(solver_2.abs_vor.to_numpy().transpose(), cmap='jet', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('2nd order Semi-Lagrangian')
    axs[0, 1].axis('off')
    # subplot 3
    axs[1, 0].imshow(solver_b.abs_vor.to_numpy().transpose(), cmap='jet', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('CF+BFECC')
    axs[1, 0].axis('off')
    # subplot 4
    axs[1, 1].imshow(solver_m.abs_vor.to_numpy().transpose(), cmap='jet', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('CF+MCM')
    axs[1, 1].axis('off')
    # unified colorbar
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    # save image
    plt.savefig(f'./output/{cur_frame}.png')
    cbar.remove()

    cur_frame += 1
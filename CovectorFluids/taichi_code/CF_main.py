import taichi as ti
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from CF_taylor import TaylorGreenVortex
from CF_leapfrog import Leapfrog
from CF_solver import CovectorFluidSolver
from CF_solver import Scheme

parser = argparse.ArgumentParser()
parser.add_argument('--demo',
                    '-d',
                    type=int,
                    default=0,
                    help='0: Taylor-Green vortex, 1: Leapfrog')
parser.add_argument('--scheme',
                    '-s',
                    type=int,
                    default=3,
                    help='1: CF+SL_1ST, 2: CF+SL_2ND, 3: CF+BFECC, 4: CF+MCM_2ND')
args, unknowns = parser.parse_known_args()

ti.init(arch=ti.gpu)

nx = 256
ny = 256
L = 2*ti.math.pi
h = L/nx
dt = 0.025
total_frame = 600 if args.demo == 0 else 2000 if args.demo == 1 else 0
cur_frame = 0

taylorVor = TaylorGreenVortex(res=nx)
leapfrog = Leapfrog(res=nx)
solver_0 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_SL_1ST)
solver_1 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_SL_2ND)
solver_2 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_BFECC)
solver_3 = CovectorFluidSolver(nx, ny, h, dt, scheme=Scheme.CF_MCM_2ND)

if args.demo == 0:
    taylorVor.generate_field()
    solver_0.init_velocity(taylorVor.velx, taylorVor.vely)
    solver_1.init_velocity(taylorVor.velx, taylorVor.vely)
    solver_2.init_velocity(taylorVor.velx, taylorVor.vely)
    solver_3.init_velocity(taylorVor.velx, taylorVor.vely)
elif args.demo == 1:
    leapfrog.generate_field()
    solver_0.init_velocity(leapfrog.velx, leapfrog.vely)
    solver_1.init_velocity(leapfrog.velx, leapfrog.vely)
    solver_2.init_velocity(leapfrog.velx, leapfrog.vely)
    solver_3.init_velocity(leapfrog.velx, leapfrog.vely)

timer = 0.0
gui = ti.GUI('Taylor Green Vortex', (nx+1, ny+1)) if args.demo == 0 \
      else ti.GUI('Leapfrog', (nx+1, ny+1)) if args.demo == 1 \
      else None
while cur_frame < total_frame:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
    if args.scheme == 0:
        solver_0.advance()
        gui.set_image(solver_0.abs_vor)
    elif args.scheme == 1:
        solver_1.advance()
        gui.set_image(solver_1.abs_vor)
    elif args.scheme == 2:
        solver_2.advance()
        gui.set_image(solver_2.abs_vor)
    elif args.scheme == 3:
        solver_3.advance()
        gui.set_image(solver_3.abs_vor)
    delta_t = time.time() - timer
    timer = time.time()
    gui.text(content=f'fps: {(1.0/delta_t):.1f}', pos=(0,0.98), color=0xffaa77)
    gui.show()
    cur_frame += 1

# vmin = 0.0
# vmax = 10.0
# while cur_frame < total_frame:
#     ## Plot in one figure
#     solver_0.advance()
#     solver_1.advance()
#     solver_2.advance()
#     solver_3.advance()
#     plt.close()
#     fig, axs = plt.subplots(2, 2, figsize=(8, 8))
#     # subplot 1
#     im1 = axs[0, 0].imshow(np.rot90(solver_0.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#     axs[0, 0].set_title('1st order Semi-Lagrangian')
#     axs[0, 0].axis('off')
#     # subplot 2
#     axs[0, 1].imshow(np.rot90(solver_1.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#     axs[0, 1].set_title('2nd order Semi-Lagrangian')
#     axs[0, 1].axis('off')
#     # subplot 3
#     axs[1, 0].imshow(np.rot90(solver_2.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#     axs[1, 0].set_title('CF+BFECC')
#     axs[1, 0].axis('off')
#     # subplot 4
#     axs[1, 1].imshow(np.rot90(solver_3.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#     axs[1, 1].set_title('CF+MCM')
#     axs[1, 1].axis('off')
#     # unified colorbar
#     cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
#     # save image
#     plt.savefig(f'./output/{cur_frame}.png')
#     cbar.remove()

#     ## Plot in seperate figures
#     plt.close()
#     if args.scheme == 0:
#         plt.imshow(np.rot90(solver_0.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#         plt.title('Semi-Lagrangian')
#         solver_0.advance()
#     elif args.scheme == 1:
#         plt.imshow(np.rot90(solver_1.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#         plt.title('2nd order Semi-Lagrangian')
#         solver_1.advance()
#     elif args.scheme == 2:
#         plt.imshow(np.rot90(solver_2.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#         plt.title('CF+BFECC')
#         solver_2.advance()
#     elif args.scheme == 3:
#         plt.imshow(np.rot90(solver_3.abs_vor.to_numpy()), cmap='jet', vmin=vmin, vmax=vmax)
#         plt.title('CF+MCM')
#         solver_3.advance()
#     plt.axis('off')
#     plt.colorbar()
#     plt.savefig(f'./output/{cur_frame}.png')

#     cur_frame += 1
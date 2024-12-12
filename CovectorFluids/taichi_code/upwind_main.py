import taichi as ti
import time
import numpy as np
import matplotlib.pyplot as plt
from CF_leapfrog import Leapfrog
from CF_taylor import TaylorGreenVortex
from upwind_solver import UpwindSolver
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--demo',
                    '-d',
                    type=int,
                    default=0,
                    help='0: Taylor-Green vortex, 1: Leapfrog')
args, unknowns = parser.parse_known_args()

ti.init(arch=ti.gpu)

nx = 256
ny = 256
L = 2*ti.math.pi
h = L/nx
dt = 0.01
total_frame = 1500 if args.demo == 0 else 5000 if args.demo == 1 else 0
cur_frame = 0

taylorVor = TaylorGreenVortex(res=nx)
leapfrog = Leapfrog(res=nx)
solver = UpwindSolver(nx, ny, h, dt)

if args.demo == 0:
    taylorVor.generate_field()
    solver.init_velocity(taylorVor.velx, taylorVor.vely)
elif args.demo == 1:
    leapfrog.generate_field()
    solver.init_velocity(leapfrog.velx, leapfrog.vely)

timer = 0.0
gui = ti.GUI('Taylor Green Vortex', (nx+1, ny+1)) if args.demo == 0 \
      else ti.GUI('Leapfrog', (nx+1, ny+1)) if args.demo == 1 \
      else None
while cur_frame < total_frame:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
    
    solver.advance()
    gui.set_image(solver.abs_vor)
    
    delta_t = time.time() - timer
    timer = time.time()
    gui.text(content=f'fps: {(1.0/delta_t):.1f}', pos=(0,0.98), color=0xffaa77)
    gui.show()
    cur_frame += 1
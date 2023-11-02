import taichi as ti
import math
import schrodinger_setting
import time

import schrodinger_simulator
import schrodinger_example
from utils.macros import *
from Visualization.experiment_ui import experiment_ui

arch_select="cpu"
if(arch_select is "gpu"):
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)

# setup example
leap_frog=schrodinger_example.LeapFrog()
leap_frog.init()
leap_frog.generate_particle()

print(leap_frog.n_grid,leap_frog.range_grid)

sim=schrodinger_simulator.SchrodingerSimulator(h_plank_leap_frog,leap_frog.n_grid,leap_frog.range_grid,100000,1.0/24,arch_select)
sim.build()
sim.set_particles(leap_frog.particle_number,leap_frog.particle_x,leap_frog.particle_v,leap_frog.particle_valid)
sim.initialize(leap_frog.grid_wave1r,leap_frog.grid_wave1i,leap_frog.grid_wave2r,leap_frog.grid_wave2i)
sim.precomputing()

ui=experiment_ui(leap_frog.range_grid,768,768)

def run():
    s=0
    while ui.window.running:
        ui.draw_setting()        
        ui.output_events()

        if(s>int(sim.total_time // sim.dt)):
            exit(0)
        
        if(~ui.pause):
            start = time.time()
            sim.update()
            end = time.time()
            
        s+=1
        print("frame "+str(s)+","+"time:"+str(s*sim.dt)+"execute time:"+str(end-start)+"/s")

        ui.draw_particles(sim.particle_x)
        ui.draw_others()        
        ui.draw_show()

run()

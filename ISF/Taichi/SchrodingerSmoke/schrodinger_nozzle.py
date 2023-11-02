import taichi as ti
import math
import schrodinger_setting

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
nozzle=schrodinger_example.Nozzle()
nozzle.init()
nozzle.generate_particle()

print(nozzle.n_grid,nozzle.range_grid)

sim=schrodinger_simulator.SchrodingerSimulator(h_plank_nozzle,nozzle.n_grid,nozzle.range_grid,100000,10.0/24,arch_select)
sim.build()
sim.set_particles(nozzle.particle_number,nozzle.particle_x,nozzle.particle_v,nozzle.particle_valid)
sim.initialize(nozzle.grid_wave1r,nozzle.grid_wave1i,nozzle.grid_wave2r,nozzle.grid_wave2i)
sim.precomputing()

ui=experiment_ui(nozzle.range_grid,768,768)

def run():
    s=0
    while ui.window.running:
        ui.draw_setting()        
        ui.output_events()

        if(s>int(sim.total_time // sim.dt)):
            exit(0)
        print("frame "+str(s)+","+"time:"+str(s*sim.dt))
        if(~ui.pause):
            sim.update()
        s+=1

        ui.draw_particles(sim.particle_x)
        ui.draw_others()        
        ui.draw_show()

run()

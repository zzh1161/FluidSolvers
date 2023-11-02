import taichi as ti
import Visualization.visualization_setting
from utils.macros import *

class experiment_ui:
    
    def __init__(
            self,
            range_grid,
            windows_width=768,
            windows_height=768
        ):
        self.windows_width,self.windows_height=windows_width,windows_height
        self.range_grid=range_grid
        self.window = ti.ui.Window("Test voxelize", (windows_width,windows_height))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()

        self.set_environment()
        self.pause=False

        self.set_x_view()

    def set_environment(self):
        # construct box
        self.box_edge=ti.Vector.field(3, dtype=ti.f32, shape=24)
        self.box_edge[0],self.box_edge[1]=vec3f_ti(0,0,0),vec3f_ti(self.range_grid[0],0,0)
        self.box_edge[2],self.box_edge[3]=vec3f_ti(0,0,0),vec3f_ti(0,self.range_grid[1],0)
        self.box_edge[4],self.box_edge[5]=vec3f_ti(0,0,0),vec3f_ti(0,0,self.range_grid[2])
        self.box_edge[6],self.box_edge[7]=vec3f_ti(self.range_grid[0],0,0),vec3f_ti(self.range_grid[0],self.range_grid[1],0)
        self.box_edge[8],self.box_edge[9]=vec3f_ti(self.range_grid[0],0,0),vec3f_ti(self.range_grid[0],0,self.range_grid[2])
        self.box_edge[10],self.box_edge[11]=vec3f_ti(0,self.range_grid[1],0),vec3f_ti(self.range_grid[0],self.range_grid[1],0)
        self.box_edge[12],self.box_edge[13]=vec3f_ti(0,self.range_grid[1],0),vec3f_ti(0,self.range_grid[1],self.range_grid[2])
        self.box_edge[14],self.box_edge[15]=vec3f_ti(0,0,self.range_grid[2]),vec3f_ti(0,self.range_grid[1],self.range_grid[2])
        self.box_edge[16],self.box_edge[17]=vec3f_ti(0,0,self.range_grid[2]),vec3f_ti(self.range_grid[0],0,self.range_grid[2])
        self.box_edge[18],self.box_edge[19]=vec3f_ti(self.range_grid[0],0,self.range_grid[2]),vec3f_ti(self.range_grid[0],self.range_grid[1],self.range_grid[2])
        self.box_edge[20],self.box_edge[21]=vec3f_ti(0,self.range_grid[1],self.range_grid[2]),vec3f_ti(self.range_grid[0],self.range_grid[1],self.range_grid[2])
        self.box_edge[22],self.box_edge[23]=vec3f_ti(self.range_grid[0],self.range_grid[1],0),vec3f_ti(self.range_grid[0],self.range_grid[1],self.range_grid[2])

        # construct axes
        self.axes_edge=ti.Vector.field(3, dtype=ti.f32, shape=6)
        self.axes_edge[0],self.axes_edge[1]=vec3f_ti(0,0,0),vec3f_ti(self.range_grid[0],0,0)
        self.axes_edge[2],self.axes_edge[3]=vec3f_ti(0,0,0),vec3f_ti(0,self.range_grid[1],0)
        self.axes_edge[4],self.axes_edge[5]=vec3f_ti(0,0,0),vec3f_ti(0,0,self.range_grid[2])

        self.axes_edge_color=ti.Vector.field(3, dtype=ti.f32, shape=6)
        self.axes_edge_color[0],self.axes_edge[1]=vec3f_ti(1,0,0),vec3f_ti(1,0,0)
        self.axes_edge_color[2],self.axes_edge[3]=vec3f_ti(0,1,0),vec3f_ti(0,1,0)
        self.axes_edge_color[4],self.axes_edge[5]=vec3f_ti(0,0,1),vec3f_ti(0,0,1)
    
    def draw_setting(self):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((1, 1, 1))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    def draw_others(self):
        self.scene.lines(self.box_edge, width=2, indices=None, color=(1, 1, 0))
        self.scene.lines(self.axes_edge, width=4, indices=None, per_vertex_color =self.axes_edge_color)
    
    def draw_particles(self,particles,color=(1,1,1),radius=0.01):
        self.scene.particles(particles, color = color, radius = radius)
    
    def output_events(self):
        self.camera.track_user_inputs(self.window,movement_speed=0.03, hold_key=ti.ui.LMB)
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == ti.ui.SPACE: 
                self.pause=~self.pause
            elif self.window.event.key == 'x':
                self.set_x_view()
            elif self.window.event.key == 'y':
                self.set_y_view()
            elif self.window.event.key == 'z':
                self.set_z_view()

    def draw_show(self):
        self.canvas.scene(self.scene)
        self.window.show()

    def set_x_view(self):
        self.camera.position(self.range_grid[0]*2,self.range_grid[1]/2,self.range_grid[2]/2)
        self.camera.lookat(0,self.range_grid[1]/2,self.range_grid[2]/2)
        self.camera.up(0,1,0)
        #camera.lookat(x, y, z)
        pass

    def set_y_view(self):
        self.camera.position(self.range_grid[0]/2,self.range_grid[1]*2,self.range_grid[2]/2)
        self.camera.lookat(self.range_grid[0]/2,0,self.range_grid[2]/2)
        self.camera.up(1,0,0)
        #camera.lookat(x, y, z)
        pass

    def set_z_view(self):
        self.camera.position(self.range_grid[0]/2,self.range_grid[1]/2,self.range_grid[2]*2)
        self.camera.lookat(self.range_grid[0]/2,self.range_grid[1]/2, 0)
        #camera.lookat(x, y, z)
        self.camera.up(0,1,0)
        pass

    

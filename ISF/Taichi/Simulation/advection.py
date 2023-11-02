	
import taichi as ti
import taichi.math as tm
import Simulation.simulation_setting

import Math.intoplation as intoplation
from utils.macros import *

class AdvectionSolver3D:
    def __init__(
            self,
            min_corner,
            dxyz
        ):
        self.min_corner=min_corner
        self.dxyz=dxyz 
    
    @ti.func
    def macGrid_semi_lagrangian_pos3D(
            self,
            dt,
            v,
            pos,
            periodical_boundary=True,
            extra_value=0
        ):

        vel0=intoplation.macGrid_face_vector_intoplation_3D(
            pos,
            self.min_corner,
            self.dxyz,    
            v,
            periodical=periodical_boundary,
            extra_value=extra_value
        )
        pos1=pos-vel0*0.5*dt
        vel1=intoplation.macGrid_face_vector_intoplation_3D(
            pos1,
            self.min_corner,
            self.dxyz,    
            v,
            periodical=periodical_boundary,
            extra_value=extra_value
        )
        return pos-vel1*dt
    
    @ti.func
    def macGrid_semi_lagrangian_center3D(
        self,
        idx,
        v,
        center_field,
        dt,
        periodical_boundary=True,
        extra_value=0
    ):
        pos=self.min_corner+(idx+0.5)*self.dxyz
        backtraced_pos=self.macGrid_semi_lagrangian_pos3D(
            dt,
            v,
            pos,
            periodical_boundary=periodical_boundary,
            extra_value=extra_value
        )
        return intoplation.macGrid_center_intoplation_3D(
            backtraced_pos,
            self.min_corner,
            self.dxyz,    
            center_field,
            periodical=periodical_boundary,
            extra_value=extra_value
        )

    @ti.func
    def macGrid_semi_lagrangian_face3D(
        self,
        idx,
        v,
        face_field,
        axis,
        dt,
        periodical_boundary=True,
        extra_value=0
    ):
        pos=self.min_corner+(idx+0.5)*self.dxyz+vec3d_ti(self.dxyz[0]*0.5,0,0)
        if(axis==1):
            pos=self.min_corner+(idx+0.5)*self.dxyz+vec3d_ti(0,self.dxyz[1]*0.5,0)
        elif(axis==2):
            pos=self.min_corner+(idx+0.5)*self.dxyz+vec3d_ti(0,0,self.dxyz[2]*0.5)
        backtraced_pos=self.macGrid_semi_lagrangian_pos3D(
            dt,
            v,
            pos,
            periodical_boundary=periodical_boundary,
            extra_value=extra_value
        )
        return intoplation.macGrid_face_intoplation_3D(
            backtraced_pos,
            self.min_corner,
            self.dxyz,    
            face_field,
            axis,
            periodical=periodical_boundary,
            extra_value=extra_value
        )
    
    @ti.func
    def macGrid_semi_lagrangian_face_vec3D(
        self,
        idx,
        v,
        face_field,
        dt,
        periodical_boundary=True,
        extra_value=0
    ):
        vx=self.macGrid_semi_lagrangian_face3D(
            idx,
            v,
            face_field,
            0,
            dt,
            periodical_boundary,
            extra_value
        )
        vy=self.macGrid_semi_lagrangian_face3D(
            idx,
            v,
            face_field,
            1,
            dt,
            periodical_boundary,
            extra_value
        )
        vz=self.macGrid_semi_lagrangian_face3D(
            idx,
            v,
            face_field,
            2,
            dt,
            periodical_boundary,
            extra_value
        )
        return vec3d_ti(vx,vy,vz)
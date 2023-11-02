import taichi as ti
import taichi.math as tm

import Math.mathn_setting
from utils.macros import *

###########################################################
#### Note: I implement Mac grid as Schrödinger’s Smoke ####
########## Only 3 dimension is available now  #############
###########################################################

@ti.func
def macGrid_face_vector_intoplation_3D(
    pos,
    min_corner,
    dxyz,    
    face_field,
    periodical=True,
    extra_value=0
):
    vec_x=macGrid_face_intoplation_3D(
        pos,
        min_corner,
        dxyz,
        face_field,
        0,
        periodical,
        extra_value
    )
    vec_y=macGrid_face_intoplation_3D(
        pos,
        min_corner,
        dxyz,
        face_field,
        1,
        periodical,
        extra_value
    )
    vec_z=macGrid_face_intoplation_3D(
        pos,
        min_corner,
        dxyz,
        face_field,
        2,
        periodical,
        extra_value
    )
    return vec3d_ti(vec_x,vec_y,vec_z)

@ti.func
def macGrid_face_intoplation_3D(
    pos,
    min_corner,
    dxyz,    
    face_field,
    axis=0, # 0 for x, 1 for y and 2 for z
    periodical=True,
    extra_value=0
):
    min_face=min_corner+0.5*dxyz+vec3d_ti(dxyz[0]*0.5,0,0)
    if(axis==1):
        min_face=min_corner+0.5*dxyz+vec3d_ti(0,dxyz[1]*0.5,0)
    elif(axis==2):
        min_face=min_corner+0.5*dxyz+vec3d_ti(0,0,dxyz[2]*0.5)
    idx=int((pos-min_face)/dxyz)
    frac=pos-idx*dxyz
    n_size=vec3d_ti(face_field.shape)

    if(periodical):
        idx=(idx+n_size)%n_size
        next_idx=(idx+1+n_size)%n_size
        res = linear_intoplation_3D(
            frac,
            face_field[idx[0],idx[1],idx[2]][axis],
            face_field[idx[0],idx[1],next_idx[2]][axis],
            face_field[idx[0],next_idx[1],idx[2]][axis],
            face_field[idx[0],next_idx[1],next_idx[2]][axis],
            face_field[next_idx[0],idx[1],idx[2]][axis],
            face_field[next_idx[0],idx[1],next_idx[2]][axis],
            face_field[next_idx[0],next_idx[1],idx[2]][axis],
            face_field[next_idx[0],next_idx[1],next_idx[2]][axis]
        )
    else:
        res = linear_intoplation_3D(
            frac,
            face_field[idx[0],idx[1],idx[2]][axis] if valid_face_idx_3D(idx,n_size) else extra_value,
            face_field[idx[0],idx[1],next_idx[2]][axis] if valid_face_idx_3D(vec3i_ti(idx[0],idx[1],next_idx[2]),n_size) else extra_value,
            face_field[idx[0],next_idx[1],idx[2]][axis] if valid_face_idx_3D(vec3i_ti(idx[0],next_idx[1],idx[2]),n_size) else extra_value,
            face_field[idx[0],next_idx[1],next_idx[2]][axis] if valid_face_idx_3D(vec3i_ti(idx[0],next_idx[1],next_idx[2]),n_size) else extra_value,
            face_field[next_idx[0],idx[1],idx[2]][axis] if valid_face_idx_3D(vec3i_ti(next_idx[0],idx[1],idx[2]),n_size) else extra_value,
            face_field[next_idx[0],idx[1],next_idx[2]][axis] if valid_face_idx_3D(vec3i_ti(next_idx[0],idx[1],next_idx[2]),n_size) else extra_value,
            face_field[next_idx[0],next_idx[1],idx[2]][axis] if valid_face_idx_3D(vec3i_ti(next_idx[0],next_idx[1],idx[2]),n_size) else extra_value,
            face_field[next_idx[0],next_idx[1],next_idx[2]][axis] if valid_face_idx_3D(vec3i_ti(next_idx[0],next_idx[1],next_idx[2]),n_size) else extra_value
        )
    return res


@ti.func
def macGrid_center_intoplation_3D(
    pos,
    min_corner,
    dxyz,                           # taichi 3D vector
    center_field:ti.template(),     # taichi field
    periodical=True,
    extra_value=0
):
    min_center=min_corner+0.5*dxyz
    idx=int((pos-min_center)/dxyz)
    frac=pos-idx*dxyz
    n_size=vec3d_ti(center_field.shape)

    if(periodical):
        idx=(idx+n_size)%n_size
        next_idx=(idx+1+n_size)%n_size
        res = linear_intoplation_3D(
            frac,
            center_field[idx[0],idx[1],idx[2]],
            center_field[idx[0],idx[1],next_idx[2]],
            center_field[idx[0],next_idx[1],idx[2]],
            center_field[idx[0],next_idx[1],next_idx[2]],
            center_field[next_idx[0],idx[1],idx[2]],
            center_field[next_idx[0],idx[1],next_idx[2]],
            center_field[next_idx[0],next_idx[1],idx[2]],
            center_field[next_idx[0],next_idx[1],next_idx[2]]
        )

    else:
        next_idx=idx+1
        res = linear_intoplation_3D(
            frac,
            center_field[idx[0],idx[1],idx[2]] if valid_center_idx_3D(idx,n_size) else extra_value,
            center_field[idx[0],idx[1],next_idx[2]] if valid_center_idx_3D(vec3i_ti(idx[0],idx[1],next_idx[2]),n_size) else extra_value,
            center_field[idx[0],next_idx[1],idx[2]] if valid_center_idx_3D(vec3i_ti(idx[0],next_idx[1],idx[2]),n_size) else extra_value,
            center_field[idx[0],next_idx[1],next_idx[2]] if valid_center_idx_3D(vec3i_ti(idx[0],next_idx[1],next_idx[2]),n_size) else extra_value,
            center_field[next_idx[0],idx[1],idx[2]] if valid_center_idx_3D(vec3i_ti(next_idx[0],idx[1],idx[2]),n_size) else extra_value,
            center_field[next_idx[0],idx[1],next_idx[2]] if valid_center_idx_3D(vec3i_ti(next_idx[0],idx[1],next_idx[2]),n_size) else extra_value,
            center_field[next_idx[0],next_idx[1],idx[2]] if valid_center_idx_3D(vec3i_ti(next_idx[0],next_idx[1],idx[2]),n_size) else extra_value,
            center_field[next_idx[0],next_idx[1],next_idx[2]] if valid_center_idx_3D(vec3i_ti(next_idx[0],next_idx[1],next_idx[2]),n_size) else extra_value
        )
    return res

@ti.func
def linear_intoplation_3D(
    frac,
    v_000,
    v_001,
    v_010,
    v_011,
    v_100,
    v_101,
    v_110,
    v_111
):
    vx_00=v_000*(1-frac[0])+v_100*frac[0]
    vx_01=v_001*(1-frac[0])+v_101*frac[0]
    vx_10=v_010*(1-frac[0])+v_110*frac[0]
    vx_11=v_011*(1-frac[0])+v_111*frac[0]

    vy0=vx_00*(1-frac[1])+vx_10*frac[1]
    vy1=vx_01*(1-frac[1])+vx_11*frac[1]

    v=vy0*(1-frac[2])+vy1*frac[2]

    return v

@ti.func
def valid_center_idx_3D(
    idx,
    n_size
):
    valid=False
    if((idx>=0).all() and (idx<n_size).all()):
        valid = True
    return valid

@ti.func
def valid_face_idx_3D(
    idx,
    n_size
):
    valid=False
    if((idx>=0).all() and (idx<n_size-1).all()):
        valid = True
    return valid    

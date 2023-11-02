########### physical constant ############
import math
h_plank=0.07#6.62607015e-34/2/math.pi

h_plank_nozzle=0.02
h_plank_leap_frog=0.1

############ dtype #############
import numpy as np
import torch
import taichi as ti
dprecision = 64
# dprecision = 64
DTYPE_TI = eval(f'ti.f{dprecision}')
DTYPE_NP = eval(f'np.float{dprecision}')
DTYPE_TC = eval(f'torch.float{dprecision}')

vec4d_ti = ti.types.vector(4, DTYPE_TI)
vec3d_ti = ti.types.vector(3, DTYPE_TI)
vec2d_ti = ti.types.vector(2, DTYPE_TI)
vec1d_ti = ti.types.vector(1, DTYPE_TI)

vec4f_ti = ti.types.vector(4, ti.f32)
vec3f_ti = ti.types.vector(3, ti.f32)
vec2f_ti = ti.types.vector(2, ti.f32)
vec1f_ti = ti.types.vector(1, ti.f32)

vec4i_ti = ti.types.vector(4, int)
vec3i_ti = ti.types.vector(3, int)
vec2i_ti = ti.types.vector(2, int)
vec1i_ti = ti.types.vector(1, int)
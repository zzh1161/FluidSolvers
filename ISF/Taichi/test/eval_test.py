import taichi as ti

def f_1():
    print("123")
def f_2():
    print("123456")

def f():
    return "f()"

a=eval(f"f")
a

ti.init()
a=ti.field(ti.f32, shape=(4,5,1))
b=a.shape
print(b)
print(type(b))

@ti.func
def f(a,b):
    return a+b


@ti.func
def g(func):
    print(func(3,4))



import copy
bb=ti.field(ti.f32, shape=(4,5,1))
bb[1,1,1]=30
aa=ti.field(ti.f32, shape=(4,5,1))
aa.copy_from(bb)
bb[1,1,1]=50
print(aa[1,1,1],bb[1,1,1])

vec4i_ti = ti.types.vector(4, int)
vec3i_ti = ti.types.vector(3, int)
vec2i_ti = ti.types.vector(2, int)
vec1i_ti = ti.types.vector(1, int)

idx=vec3i_ti(1,1,1)
print(aa[idx[0],idx[1],idx[2]])

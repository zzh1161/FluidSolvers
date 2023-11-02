import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu)

@ti.func
def f(z):
    return z[0]+z[1]

@ti.kernel
def test():
    for i in range(10):
        print(f((1,2)))
    
class A:
    def __init__(self):
        self.TTT=100

class B(A):
    def __init(self):
        super().__init__()

b=B()
print(b.TTT)

x=ti.Vector.field(3, dtype=float, shape=10)
x[1]=(10,10,10)
x[2]=(2,3,4)
print(x[1])
print(x[1]+1)
print((x[1]+1)%10)
print(x[1]/x[2])
print(x[1]//x[2])
a=x[1]>x[0]
print(a.all())

print(x[0])
@ti.kernel
def test_field(v:ti.template()):
    v[0]=(1213,1313,13131)

test_field(x)
print(x[0])

vec3d_ti = ti.types.vector(3, float)
a=vec3d_ti([1,2,5])
print(a)

import numpy as np
x=np.array([[1+1j,1+2j],[1.1+1.9j,3+1j]])
print(x)



vec3i_ti = ti.types.vector(3, int)
x=vec3i_ti(1,6,4)
print(x^61)

@ti.kernel
def f():
    x=vec3d_ti(1,6,4)
    x=tm.normalize(x)
    print(x)

f()
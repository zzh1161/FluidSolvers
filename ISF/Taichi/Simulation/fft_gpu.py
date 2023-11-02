# adapted from:
# https://www.idtools.com.au/gpu-accelerated-fft-compatible-with-numpy/

# how to get fft for complex
# https://fzhhzf.wordpress.com/2011/03/14/实序列快速傅里叶变换rfft和共轭对称矩阵逆变换/
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft

# a strange thing: plan_forward of same size cannot be create several times in the fuction fft2_gpu
plan_forward_list={}
plan_backward_list={}

def fft2_cpu(x, fftshift=False):
    if fftshift is False:
        res=np.fft.fft2(x)
    else:
        res=np.fft.fftshift(np.fft.fft2(x))
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def fft2_complex_cpu(x,y,fftshift=False):
    z=np.asarray(x, np.complex64)+np.asarray(y, np.complex64)*1j
    if fftshift is False:
        res = np.fft.fft2(z)
    else:
        res = np.fft.fftshift(np.fft.fft2(z))
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def fft2_gpu(x, fftshift=False):
    global plan_forward_list

    # Convert the input array to single precision float
    if x.dtype != 'float32':
        x = x.astype('float32')

    # Get the shape of the initial numpy array
    n1, n2= x.shape
    
    # From numpy array to GPUarray
    xgpu = gpuarray.to_gpu(x)
    
    # Initialise output GPUarray 
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal.
    y = gpuarray.empty((n1,n2//2 + 1), np.complex64)
    
    # Forward FFT
    
    plan_forward_name= str(n1)+"#"+str(n2)
    if(plan_forward_name not in plan_forward_list):
        plan_forward_list[plan_forward_name]=cu_fft.Plan((n1, n2), np.float32, np.complex64)
    plan_forward=plan_forward_list[plan_forward_name]

    cu_fft.fft(xgpu, y, plan_forward)
    
    left = y.get()

    # To make the output array compatible with the numpy output
    # we need to stack horizontally the y.get() array and its flipped version
    # We must take care of handling even or odd sized array to get the correct 
    # size of the final array   
    if n2//2 == n2/2:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,1:-1],1,axis=0)
    else:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,:-1],1,axis=0) 
    
    # Get a numpy array back compatible with np.fft
    if fftshift is False:
        yout = np.hstack((left,np.conjugate(right)))
    else:
        yout = np.fft.fftshift(np.hstack((left,np.conjugate(right))))

    res=yout.astype('complex128')
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def fft2_complex_gpu(x,y, fftshift=False):
    global plan_forward_list
    z=x+y*1j
    # Convert the input array to single precision float
    if z.dtype != 'complex64':
        z = z.astype('complex64')

    # Get the shape of the initial numpy array
    n1, n2= z.shape
    
    # From numpy array to GPUarray
    zgpu = gpuarray.to_gpu(z)
    
    # Initialise output GPUarray 
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal.
    res_gpu = gpuarray.empty((n1,n2), np.complex64)
    
    # Forward FFT
    
    plan_forward_name= str(n1)+"#"+str(n2)+"c"
    if(plan_forward_name not in plan_forward_list):
        plan_forward_list[plan_forward_name]=cu_fft.Plan((n1, n2), np.complex64, np.complex64)
    plan_forward=plan_forward_list[plan_forward_name]

    cu_fft.fft(zgpu, res_gpu, plan_forward)
    
    # Get a numpy array back compatible with np.fft
    if fftshift is False:
        yout = res_gpu.get()
    else:
        yout = np.fft.fftshift(res_gpu.get())

    res=yout.astype('complex128')
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def fft3_cpu(x, fftshift=False):
    if fftshift is False:
        res = np.fft.fftn(x)
    else:
        res = np.fft.fftshift(np.fft.fftn(x))
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def fft3_complex_cpu(x,y, fftshift=False):
    z=np.asarray(x, np.complex64)+np.asarray(y, np.complex64)*1j
    if fftshift is False:
        res = np.fft.fftn(z)
    else:
        res = np.fft.fftshift(np.fft.fftn(z))
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def fft3_gpu(x, fftshift=False):
    global plan_forward_list

    # Convert the input array to single precision float
    if x.dtype != 'float32':
        x = x.astype('float32')

    # Get the shape of the initial numpy array
    n1, n2, n3= x.shape
    
    # From numpy array to GPUarray
    xgpu = gpuarray.to_gpu(x)
    
    # Initialise output GPUarray 
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal.
    y = gpuarray.empty((n1,n2,n3//2 + 1), np.complex64)
    
    # Forward FFT
    
    plan_forward_name= str(n1)+"#"+str(n2)+"#"+str(n3)
    if(plan_forward_name not in plan_forward_list):
        plan_forward_list[plan_forward_name]=cu_fft.Plan((n1, n2,n3), np.float32, np.complex64)
    plan_forward=plan_forward_list[plan_forward_name]

    cu_fft.fft(xgpu, y, plan_forward)
    
    left = y.get()

    # To make the output array compatible with the numpy output
    # we need to stack horizontally the y.get() array and its flipped version
    # We must take care of handling even or odd sized array to get the correct 
    # size of the final array   
    if n3//2 == n3/2:
        right = np.roll(np.flip(np.flip(y.get(),axis=1),axis=2)[:,:,1:-1],1,axis=1)
    else:
        right = np.roll(np.flip(np.flip(y.get(),axis=1),axis=2)[:,:,:-1],1,axis=1) 
    
    # Get a numpy array back compatible with np.fft
    if fftshift is False:
        yout = np.concatenate((left,np.conjugate(right)), axis=2)
        #yout = np.hstack((left,right))
    else:
        yout = np.fft.fftshift(np.concatenate((left,np.conjugate(right)), axis=2))

    res=yout.astype('complex128')

    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def fft3_complex_gpu(x,y, fftshift=False):

    global plan_forward_list

    z=x+y*1j
    if z.dtype != 'complex64':
        z = z.astype('complex64')
    # Get the shape of the initial numpy array
    n1, n2, n3= z.shape
    
    # From numpy array to GPUarray
    z_gpu = gpuarray.to_gpu(z)
    
    # Initialise output GPUarray 
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal.
    res_gpu = gpuarray.empty((n1,n2,n3), np.complex64)
    
    # Forward FFT
    
    plan_forward_name= str(n1)+"#"+str(n2)+"#"+str(n3)+"c"
    if(plan_forward_name not in plan_forward_list):
        plan_forward_list[plan_forward_name]=cu_fft.Plan((n1, n2,n3), np.complex64, np.complex64)
    plan_forward=plan_forward_list[plan_forward_name]

    cu_fft.fft(z_gpu, res_gpu, plan_forward)
    
    left = res_gpu.get()

    if fftshift is False:
        yout = res_gpu.get()
        #yout = np.hstack((left,right))
    else:
        yout = np.fft.fftshift(res_gpu.get())

    res=yout.astype('complex128')

    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))
    """
    xr,xi = fft3_gpu(x,False)
    yr,yi = fft3_gpu(y,False)
    if fftshift is False:
        res = xr+xi*1j+(yr+yi*1j)*1j
        #yout = np.hstack((left,right))
    else:
        res = np.fft.fftshift(xr+xi*1j+(yr+yi*1j)*1j)

    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))
    """ 
def ifft2_cpu(x,y, fftshift=False):
    if fftshift is False:
        res = np.real(np.fft.ifft2(x+y*1j))
    else:
        res = np.real(np.fft.ifft2(np.fft.ifftshift(x+y*1j)))
    return np.ascontiguousarray(res)

def ifft2_complex_cpu(x,y, fftshift=False):
    if fftshift is False:
        res=np.fft.ifft2(x+y*1j)
    else:
        res=np.fft.ifft2(np.fft.ifftshift(x+y*1j))
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def ifft2_gpu(x,y, fftshift=False):
    global plan_backward_list
    ''' This function produce an output that is 
    compatible with numpy.fft.ifft2
    The input y is a 2D complex numpy array'''
 
    # Get the shape of the initial numpy array
    z=x+y*1j
    n1, n2 = z.shape
    
    # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
    if fftshift is False:
        y2 = np.asarray(z[:,0:n2//2 + 1], np.complex64)
    else:
        y2 = np.asarray(np.fft.ifftshift(z)[:,:n2//2+1], np.complex64)
    ygpu = gpuarray.to_gpu(y2) 
     
    # Initialise empty output GPUarray 
    res_gpu = gpuarray.empty((n1,n2), np.float32)
    
    # Inverse FFT
    plan_backward_name= str(n1)+"#"+str(n2)
    if(plan_backward_name not in plan_backward_list):
        plan_backward_list[plan_backward_name]=cu_fft.Plan((n1, n2), np.complex64, np.float32)
    plan_backward=plan_backward_list[plan_backward_name]

    cu_fft.ifft(ygpu, res_gpu, plan_backward)
    
    # Must divide by the total number of pixels in the image to get the normalisation right
    res = res_gpu.get()/n1/n2
    
    return np.ascontiguousarray(res)

def ifft2_complex_gpu(x,y, fftshift=False):
    global plan_backward_list
    ''' This function produce an output that is 
    compatible with numpy.fft.ifft2
    The input y is a 2D complex numpy array'''
 
    # Get the shape of the initial numpy array
    z=x+y*1j
    n1, n2 = z.shape
    
    # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
    if fftshift is False:
        y2 = np.asarray(z, np.complex64)
    else:
        y2 = np.asarray(np.fft.ifftshift(z), np.complex64)
    ygpu = gpuarray.to_gpu(y2) 
     
    # Initialise empty output GPUarray 
    res_gpu = gpuarray.empty((n1,n2), np.complex64)
    
    # Inverse FFT
    plan_backward_name= str(n1)+"#"+str(n2)+"C"
    if(plan_backward_name not in plan_backward_list):
        plan_backward_list[plan_backward_name]=cu_fft.Plan((n1, n2), np.complex64, np.complex64)
    plan_backward=plan_backward_list[plan_backward_name]

    cu_fft.ifft(ygpu, res_gpu, plan_backward)
    
    # Must divide by the total number of pixels in the image to get the normalisation right
    res = res_gpu.get()/n1/n2
    
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def ifft3_cpu(x,y, fftshift=False):
    if fftshift is False:
        res = np.real(np.fft.ifftn(x+y*1j))
    else:
        res = np.real(np.fft.ifftn(np.fft.ifftshift(x+y*1j)))
    return np.ascontiguousarray(res)

def ifft3_complex_cpu(x,y, fftshift=False):
    if fftshift is False:
        res=np.fft.ifftn(x+y*1j)
    else:
        res=np.fft.ifftn(np.fft.ifftshift(x+y*1j))
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))

def ifft3_gpu(x,y, fftshift=False):
    global plan_backward_list

    # Get the shape of the initial numpy array
    z=x+y*1j
    n1, n2, n3 = z.shape
    
    # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
    if fftshift is False:
        y2 = np.asarray(z[:,:,0:n3//2 + 1], np.complex64)
    else:
        y2 = np.asarray(np.fft.ifftshift(z)[:,:,:n3//2+1], np.complex64)
    ygpu = gpuarray.to_gpu(y2) 
     
    # Initialise empty output GPUarray 
    res_gpu = gpuarray.empty((n1,n2,n3), np.float32)
    
    # Inverse FFT
    plan_backward_name= str(n1)+"#"+str(n2)+"#"+str(n3)
    if(plan_backward_name not in plan_backward_list):
        plan_backward_list[plan_backward_name]=cu_fft.Plan((n1, n2,n3), np.complex64, np.float32)
    plan_backward=plan_backward_list[plan_backward_name]

    cu_fft.ifft(ygpu, res_gpu, plan_backward)
    
    # Must divide by the total number of pixels in the image to get the normalisation right
    res = res_gpu.get()/n1/n2/n3
    
    return np.ascontiguousarray(res)

def ifft3_complex_gpu(x,y, fftshift=False):
    global plan_backward_list

    # Get the shape of the initial numpy array
    z=x+y*1j
    n1, n2, n3 = z.shape
    
    # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
    if fftshift is False:
        y2 = np.asarray(z, np.complex64)
    else:
        y2 = np.asarray(np.fft.ifftshift(z), np.complex64)
    ygpu = gpuarray.to_gpu(y2) 
     
    # Initialise empty output GPUarray 
    res_gpu = gpuarray.empty((n1,n2,n3), np.complex64)
    
    # Inverse FFT
    plan_backward_name= str(n1)+"#"+str(n2)+"#"+str(n3)+"C"
    if(plan_backward_name not in plan_backward_list):
        plan_backward_list[plan_backward_name]=cu_fft.Plan((n1, n2,n3), np.complex64, np.complex64)
    plan_backward=plan_backward_list[plan_backward_name]

    cu_fft.ifft(ygpu, res_gpu, plan_backward)
    
    # Must divide by the total number of pixels in the image to get the normalisation right
    res = res_gpu.get()/n1/n2/n3
    
    return np.ascontiguousarray(np.real(res)),np.ascontiguousarray(np.imag(res))



if __name__== "__main__":
    # compare with np.fft.fftshift of 2D matrix
    print("====================================================================")
    print("TEST 2D")
    print("====================================================================")

    arr2D=np.array([[61,12,43,54,51],[61,47,81,39,150],[171,1.2,113,104,15]],dtype="float32")
    print("2D matrix",arr2D)
    
    fft1 = fft2_cpu(arr2D, fftshift=False)
    fft2 = fft2_gpu(arr2D, fftshift=False)
    print("2D matrix in CPU with shift",fft1)
    print("2D matrix in GPU with shift",fft2)

    ifft1 = ifft2_cpu(fft1[0],fft1[1],fftshift=False)
    ifft2 = ifft2_gpu(fft2[0],fft2[1], fftshift=False)
    print("inverse 2D matrix in CPU with shift",ifft1)
    print("inverse 2D matrix in GPU with shift",ifft2)

    arr2D2=np.array([[1,21,33,41],[65,72,43,91],[1.1,112,133,1.4]],dtype="float32")
    fft1 = fft2_cpu(arr2D2, fftshift=False)
    fft2 = fft2_gpu(arr2D2, fftshift=False)
    print("2D matrix in CPU without shift",fft1)
    print("2D matrix in GPU without shift",fft2)

    ifft1 = ifft2_cpu(fft1[0],fft1[1], fftshift=False)
    ifft2 = ifft2_gpu(fft2[0],fft2[1], fftshift=False)
    print("inverse 2D matrix in CPU without shift",ifft1)
    print("inverse 2D matrix in GPU without shift",ifft2)

    # compare with np.fft.fftshift of 3D matrix
    print("====================================================================")
    print("TEST 3D")
    print("====================================================================")
    arr3D=np.array([[[15,21,33,41,500],[0.6,47,85,91,110],[141,112,1.3,114,1115]],[[1.1,1332,113,134,1145],[1146,147,1.8,119,1410],[1.11,1112,111.3,114,1.15]]],dtype="float32")
    print("3D matrix",arr3D)

    fft1 = fft3_cpu(arr3D, fftshift=True)
    fft2 = fft3_gpu(arr3D, fftshift=True)
    print("3D matrix in CPU with shift",fft1)
    print("3D matrix in GPU with shift",fft2)
    ifft1 = ifft3_cpu(fft1[0],fft1[1], fftshift=True)
    ifft2 = ifft3_gpu(fft2[0],fft2[1], fftshift=True)
    print("inverse 3D matrix in CPU with shift",ifft1)
    print("inverse 3D matrix in GPU with shift",ifft2)

    arr3D2=np.array([[[11,52,32,44],[16,75,68,69],[111,142,113,614]],[[191,142,1.3,1.4],[136,167,118,149],[1111,1412,1.13,11.4]]],dtype="float32")
    fft1 = fft3_cpu(arr3D2, fftshift=False)
    fft2 = fft3_gpu(arr3D2, fftshift=False)
    print("3D matrix in CPU without shift",fft1)
    print("3D matrix in GPU without shift",fft2)
    ifft1 = ifft3_cpu(fft1[0],fft1[1], fftshift=False)
    ifft2 = ifft3_gpu(fft2[0],fft2[1], fftshift=False)
    print("inverse 3D matrix in CPU without shift",ifft1)
    print("inverse 3D matrix in GPU without shift",ifft2)

    arr2D=np.array([[61+1j,12+2j,43+1.1j,54+1.5j,51+1.9j],[61+1.9j,47,81+1.9j,39+1.9j,150],[171+1.9j,1.2,113,104,15+1.9j]],dtype=np.complex64)
    fft1 = fft2_complex_cpu(np.real(arr2D),np.imag(arr2D), fftshift=True)
    fft2 = fft2_complex_gpu(np.real(arr2D),np.imag(arr2D), fftshift=True)
    print("2D matrix in CPU with shift for complex field",fft1)
    print("2D matrix in GPU with shift for complex field",fft2)
    ifft1x,ifft1y = ifft2_complex_cpu(fft1[0],fft1[1], fftshift=True)
    ifft2x,ifft2y = ifft2_complex_gpu(fft2[0],fft2[1], fftshift=True)
    print("inverse 2D matrix in CPU with shift for complex field",ifft1x,ifft1y)
    print("inverse 2D matrix in GPU with shift for complex field",ifft2x,ifft2y)

    # compare with np.fft.fftshift of 2D matrix for complex

    fft1 = fft2_complex_cpu(np.real(arr2D),np.imag(arr2D), fftshift=False)
    fft2 = fft2_complex_gpu(np.real(arr2D),np.imag(arr2D), fftshift=False)
    print("2D matrix in CPU without shift for complex field",fft1)
    print("2D matrix in GPU without shift for complex field",fft2)
    ifft1x,ifft1y = ifft2_complex_cpu(fft1[0],fft1[1], fftshift=False)
    ifft2x,ifft2y = ifft2_complex_gpu(fft2[0],fft2[1], fftshift=False)
    print("inverse 2D matrix in CPU without shift for complex field",ifft1x,ifft1y)
    print("inverse 2D matrix in GPU without shift for complex field",ifft2x,ifft2y)

    arr3D2=np.array([[[11+1.9j,52+5j,32+2j,44+0.9j],[16+0.9j,75,68+0.9j,69],[111+0.9j,142,113,614+0.9j]],[[191,142+0.9j,1.3,1.4],[136,167+0.9j,118+0.9j,149],[1111+0.9j,1412+0.9j,1.13,11.4]]],dtype=np.complex64)
    fft1 = fft3_complex_cpu(np.real(arr3D2),np.imag(arr3D2), fftshift=False)
    fft2 = fft3_complex_gpu(np.real(arr3D2),np.imag(arr3D2), fftshift=False)
    print("3D matrix in CPU without shift for complex field",fft1)
    print("3D matrix in GPU without shift for complex field",fft2)
    ifft1x,ifft1y = ifft3_complex_cpu(fft1[0],fft1[1], fftshift=False)
    ifft2x,ifft2y = ifft3_complex_gpu(fft2[0],fft2[1], fftshift=False)
    print("inverse 3D matrix in CPU without shift for complex field",ifft1x,ifft1y)
    print("inverse 3D matrix in GPU without shift for complex field",ifft2x,ifft2y)

    fft1 = fft3_complex_cpu(np.real(arr3D2),np.imag(arr3D2), fftshift=True)
    fft2 = fft3_complex_gpu(np.real(arr3D2),np.imag(arr3D2), fftshift=True)
    print("3D matrix in CPU with shift for complex field",fft1)
    print("3D matrix in GPU with shift for complex field",fft2)
    ifft1x,ifft1y = ifft3_complex_cpu(fft1[0],fft1[1], fftshift=True)
    ifft2x,ifft2y = ifft3_complex_gpu(fft2[0],fft2[1], fftshift=True)
    print("inverse 3D matrix in CPU with shift for complex field",ifft1x,ifft1y)
    print("inverse 3D matrix in GPU with shift for complex field",ifft2x,ifft2y)


    

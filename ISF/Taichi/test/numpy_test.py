import numpy as np
 
arr1 = np.array([[[1,3], [2,4] ],[[11,13], [12,14] ]])
arr2 = np.array([[[1,4], [2,6] ],[[11,14], [12,16] ]])

res=np.concatenate((arr1, arr2), axis=2)
# res = np.hstack((arr1, arr2))

print(arr1)
print(arr2)
print (res)

res2=np.flip(arr1, axis=1) #ud
res3=np.flip(arr1, axis=2) #lr

print(arr1)
print(res2)
print(res3)

arr1 = np.array([[1,3,9], [2,4,8]])
arr2=np.roll(arr1,1,axis=0)
print(arr1)
print(arr2)

arr1 = np.array([[[1,3], [2,4] ],[[11,13], [12,14] ]])
arr2=np.roll(arr1,1,axis=1)
print(arr1)
print(arr2)

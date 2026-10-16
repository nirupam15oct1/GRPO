import numpy as np 
x=np.array([-3.0,1.0,2.0,4.0])

if np.max(np.abs(x)) !=0:
    scale=127.0/np.max(np.abs(x))
else:
    scale=1
x_quantized=np.round(x*scale).astype(np.int8)
x_dequantized=x_quantized.astype(np.float32)/scale

print("Original Array:   ",x)
print("Quantized Array:   ", x_quantized )
print("Dequantized Array  ", x_dequantized )

mse=np.mean((x-x_dequantized)**2)
print("Mean Sqared Error:" ,mse)


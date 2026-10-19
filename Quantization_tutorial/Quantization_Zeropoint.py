import numpy as np

x = np.array([-3.0, 1.0, 2.0, 4.0])

# Integer range for int8
qmin, qmax = -128, 127

# Include 0 in the range so that real 0.0 maps exactly to an integer
r_min = min(x.min(), 0.0)
r_max = max(x.max(), 0.0)

# Scale maps the float range onto the integer range
if r_max != r_min:
    scale = (r_max - r_min) / (qmax - qmin)
else:
    scale = 1.0

# Zero point is the integer that represents real 0.0
zero_point = int(np.clip(np.round(qmin - r_min / scale), qmin, qmax))

# Quantize: q = clip(round(x / S) + Z, qmin, qmax)
x_quantized = np.clip(np.round(x / scale) + zero_point, qmin, qmax).astype(np.int8)

# Dequantize: x_hat = (q - Z) * S
x_dequantized = (x_quantized.astype(np.float32) - zero_point) * scale

print("Original Array:   ", x)
print("Quantized Array:   ", x_quantized)
print("Dequantized Array  ", x_dequantized)
print("Scale:             ", scale)
print("Zero Point:        ", zero_point)

mse = np.mean((x - x_dequantized) ** 2)
print("Mean Squared Error:", mse)

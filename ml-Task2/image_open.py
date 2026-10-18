from PIL  import Image
import numpy as np  
import torch

# import png
img=Image.open("./png/outer1.png")

print(type(img))
print(img.size)
print(img.mode)  

# numpy
img_np=np.array(img)

print(type(img_np))
print(img_np.shape)
print(img_np.dtype)

# max min  
print("max:",img_np.max())
print("min:",img_np.min()) 

# [100,100]
print(img_np[100,100])  

img_tensor=torch.tensor(img_np)
print(img_tensor.shape)
print(img_tensor.dtype)

import torch
import numpy as npy


xxx = npy.asarray([x for x in range(20)]).reshape(2,2,5)
a = torch.from_numpy(xxx)
print(a)
b = a.unbind(2)
print(b)
print(b[0])

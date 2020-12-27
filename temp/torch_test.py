import torch
import numpy as np

a = torch.tensor(np.arange(12).reshape(3, 4), dtype=torch.float32)
b = a[:, :2]
print(b)
c = a[:, :2].contiguous()
print(c)
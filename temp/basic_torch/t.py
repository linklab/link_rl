import torch

a = torch.ones((10, 2))
b = torch.zeros((10, 1))

c = a * b
print(c)
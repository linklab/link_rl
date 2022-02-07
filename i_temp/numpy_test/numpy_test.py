import numpy as np
import torch

print(np.random.randint(low=10, high=100), "!!!")

a = np.zeros(10)
b = np.array(a)
c = torch.tensor(b)

print(a)
print(b)
print(c)
print(a is b)
print(c is b)

a[0] = 10
b[9] = 1000

print(a)
print(b)
print(c)

b = np.array([[0, 0]])
b[0] = np.array([1])
print(b)
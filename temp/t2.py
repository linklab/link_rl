import torch

a = torch.randn(2, 3)
print(a)
b = torch.diag_embed(a)
print(b.size())
print(b)

a = torch.ones(4, 3)
b = torch.ones(2, 4, 3) * 2
print('a:', a)
print('b:', b)

c = a.expand_as(b)
print('c:', c)
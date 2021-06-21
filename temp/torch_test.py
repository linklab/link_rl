import torch
from torchviz import make_dot

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
print(x.shape, y.shape)

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b

print('Gradient function for z =', z.grad_fn)

loss = torch.mean(z - y) + 10000000000

print('Gradient function for loss =', loss.grad_fn)

dot = make_dot(loss, params={"w": w, "b": b})
dot.render()

loss.backward()
print(w.grad)
print(b.grad)
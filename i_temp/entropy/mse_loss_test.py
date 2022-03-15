import torch
import torch.nn.functional as F

logit = torch.Tensor(
    [[-0.3],
     [1.5],
     [0.7]]
)
y = torch.Tensor(
    [[1.0],
     [0.0],
     [1.0]]
)

# OPTION I
loss = torch.mean(torch.square(logit - y))
print(loss);print("*" * 100)

# OPTION II
loss = F.mse_loss(input=logit, target=y)
print(loss);print("*" * 100)


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
logit_sigmoid = torch.sigmoid(logit)
print(logit_sigmoid)

loss = -1.0 * torch.sum(y * torch.log(logit_sigmoid) + (1 - y) * torch.log(1 - logit_sigmoid))
print(loss);print("*" * 100)

# OPTION II
loss = F.binary_cross_entropy_with_logits(input=logit, target=y, reduction="sum")
print(loss);print("*" * 100)

# OPTION III
loss = F.binary_cross_entropy(input=logit_sigmoid, target=y, reduction="sum")
print(loss);print("*" * 100)

loss = F.mse_loss(input=logit, )
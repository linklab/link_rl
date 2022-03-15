import torch
import torch.nn.functional as F

logit = torch.Tensor(
    [[0.1, -2.3, 3.1],
     [1.5, 0.3, -1.1]]
)
y = torch.Tensor(
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0]]
)

# OPTION I
logit_softmax = torch.softmax(logit, dim=-1)
print(logit_softmax)

logit_log_softmax = torch.log(logit_softmax)
print(logit_log_softmax)

loss = -1.0 * torch.sum(torch.sum(y * logit_log_softmax, dim=-1), dim=-1)
print(loss);print("*" * 100)

# OPTION II
logit_log_softmax = torch.log_softmax(logit, dim=-1)
print(logit_log_softmax)

loss = -1.0 * torch.sum(torch.sum(y * logit_log_softmax, dim=-1), dim=-1)
print(loss);print("*" * 100)

# OPTION III
loss = F.cross_entropy(input=logit, target=y, reduction="sum")
print(loss);print("*" * 100)


import os

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

input_size = 10
hidden_size = 30
num_actions = 4
batch_size = 1
length = 1
num_layers = 3
bidirectional = True

if bidirectional:
    num_directions = 2
else:
    num_directions = 1

rnn = nn.GRU(
    input_size=input_size, hidden_size=hidden_size,
    num_layers=num_layers, bias=True, batch_first=True, bidirectional=bidirectional
)
linear = nn.Linear(in_features=hidden_size * num_directions, out_features=num_actions)

input = Variable(torch.randn(batch_size, length, input_size)) # B, T, D
hidden = Variable(torch.zeros(num_layers * num_directions, batch_size, hidden_size)) # (num_layers * num_directions, batch, hidden_size)
#cell = Variable(torch.zeros(num_layers * num_directions, batch_size, hidden_size))   # (num_layers * num_directions, batch, hidden_size)

output, hidden = rnn(input, hidden)

print(output.size())
print(hidden.size())

actions = F.softmax(linear(output), dim=1)
print(actions)
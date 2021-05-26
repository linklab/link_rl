import torch
import numpy as np
import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

# a = torch.tensor(np.arange(12).reshape(3, 4), dtype=torch.float32)
# b = a[:, :2]
# print(b)
# c = a[:, :2].contiguous()
# print(c)
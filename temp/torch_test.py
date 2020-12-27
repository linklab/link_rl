import torch
import numpy as np
import os
import sys
from pathlib import Path
PROJECT_HOME = os.path.dirname(Path(__file__).parent)

print(os.path.dirname(sys.modules['__main__'].__file__))
print(ROOT_DIR)
# a = torch.tensor(np.arange(12).reshape(3, 4), dtype=torch.float32)
# b = a[:, :2]
# print(b)
# c = a[:, :2].contiguous()
# print(c)
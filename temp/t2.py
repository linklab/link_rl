from collections import deque

import numpy as np

l = deque(maxlen=2)

l.append(10)
l.append(20)

print(l[0], l[1])
print(l)

l.append(30)

print(l[0], l[1])
print(l)
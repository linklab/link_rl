import math
import numpy as np
import matplotlib.pyplot as plt

hx = []

left = 0
right = 1
delta = right - left
points = 4096
step = delta/points
p = 0
q = 0

X = np.linspace(0, 1, points)

for i in range(1, points+1):
    p = left + i*step
    q = 1 - p
    if p >= 1:
        p = p - step
    if q <= 0:
        q = q + step

    elem1 = p * math.log(p, 2)
    elem2 = q * math.log(q, 2)
    entropy = -(elem1 + elem2)
    hx.append(entropy)

plt.ylim(-0.005, 1.05)
plt.xlim(-0.005, 1.005)
plt.yticks(np.arange(0.0, 1.05, step=0.1))
plt.xticks(np.arange(0.0, 1.005, step=0.1))

plt.xlabel('p - probability of heads (of a coin flip)')
plt.ylabel('Entropy H(p):  [0, 1]')
plt.plot(X,  hx, c = 'b', linestyle='-')

plt.grid(b=None, which='major', axis='both', linestyle='--', linewidth=1)
plt.show()

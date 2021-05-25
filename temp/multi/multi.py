import multiprocessing as mp
import numpy as np

def square(x):
    return np.square(x)

x = np.arange(64)
print(x)
print(mp.cpu_count())

pool = mp.Pool(32)
squared = pool.map(square, [x[2*i:2*i+2] for i in range(32)])

for idx, squared in enumerate(squared):
    print(idx, squared)
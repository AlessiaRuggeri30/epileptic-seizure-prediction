import numpy as np

with np.load('dataset/TWH056_Day-504_Clip-0-1.npz') as data:
    data = dict(data)

k = list(data.keys())
for key in k: print(key)
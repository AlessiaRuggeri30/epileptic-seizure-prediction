import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# sns.set_style('whitegrid')

''' Load dataset '''
# path = "../.spektral/datasets/ieeg/TWH056_Day-504_Clip-0-1.npz"     # server
path = "../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

with np.load(path) as data:
    data = dict(data)

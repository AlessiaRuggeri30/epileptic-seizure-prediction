import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# sns.set_style('whitegrid')

path = "../.spektral/datasets/ieeg/TWH056_Day-504_Clip-0-1.npz"
# path = "dataset/TWH056_Day-504_Clip-0-1.npz"

''' Load dataset '''
with np.load(path) as data:
    data = dict(data)

''' Histogram of number of True/False labels '''
szr_bool_str = [item for item in data['szr_bool'].astype(str)]
plt.hist(szr_bool_str)
plt.show()

''' Convert np array in dataframe and transpose '''
ieeg = pd.DataFrame(data['ieeg'])
ieeg = ieeg.T

electrodes = 15
timestamps = 10000
seizure_start = 1670000     # 1684381
seizure_end = 1800000       # 1699381

# ''' Correlation heatmaps of initial timestamps'''
# corr1 = ieeg[0:100].corr()
# plt.imshow(corr1, cmap="RdBu_r")
# plt.title("First 100")
# plt.show()

# ''' Correlation heatmaps of seizure timestamps'''
# corr2 = ieeg[seizure_start:seizure_start+100].corr()
# plt.imshow(corr2, cmap="RdBu_r")
# plt.title("First 100 of seizure")
# plt.show()


''' Plot of first electrodes in first timestamps'''
# ieeg_subset = ieeg.loc[0:(timestamps-1), 0:(electrodes-1)]
#
# fig1 = sns.lineplot(data=ieeg_subset, hue='szr_bool', dashes=False, legend=False)
# plt.xlabel("Time")
# plt.ylabel("Signal values")
# plt.title(f"EEG of first {electrodes} electrodes for first {timestamps} timestamps")
# plt.show()


''' Plot of first electrodes during seizure timestamps'''
# ieeg_subset = ieeg.loc[seizure_start:seizure_end, 0:(electrodes-1)]
#
# fig2 = sns.lineplot(data=ieeg_subset, hue='szr_bool', dashes=False, legend=False)
# plt.xlabel("Time")
# plt.ylabel("Signal values")
# plt.title(f"EEG of first {electrodes} electrodes during seizure timestamps")
# plt.axvline(x=1684381)
# plt.axvline(x=1699381)
# plt.show()

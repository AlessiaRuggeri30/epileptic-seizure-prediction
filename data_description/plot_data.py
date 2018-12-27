import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_clip = 3
ieeg = {}
for c in range(1, n_clip+1):

    ''' Load dataset '''
    path = f"/home/phait/datasets/ieeg/TWH056_Day-504_Clip-0-{c}.npz"     # server
    # path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

    with np.load(path) as data:
        temp = dict(data)
        ieeg[c] = (temp['ieeg']).T

ieeg_norm = ieeg
for i in range(0, 1800000):
    ieeg_norm[1][i] = ieeg_norm[1][i] - np.mean(ieeg_norm[1][i])

value = 150
for i in range(1, 1800000+1):
    ieeg_norm[1][i-1] = ieeg_norm[1][i-1] + (i*value)

electrodes = 15
timestamps = 10000
seizure_start = 1670000     # 1684381
seizure_end = 1800000       # 1699381

''' Correlation heatmaps of initial timestamps'''
# corr1 = ieeg[0:100].corr()
# plt.imshow(corr1, cmap="RdBu_r")
# plt.title("First 100")
# plt.show()

# ''' Correlation heatmaps of seizure timestamps'''
# corr2 = ieeg[seizure_start:seizure_start+100].corr()
# plt.imshow(corr2, cmap="RdBu_r")
# plt.title("First 100 of seizure")
# plt.show()


# figure1 = plt.figure(figsize=(10.0, 6.0))
# plt.plot(ieeg[1])
# plt.xlabel("Time")
# plt.ylabel("Signal values")
# plt.title(f"IEEG")
# plt.savefig("seizure1.png", dpi=400)

figure2 = plt.figure(figsize=(10.0, 6.0))
plt.plot(ieeg_norm[1])
plt.xlabel("Time")
plt.ylabel("Signal values")
plt.title(f"IEEG")
plt.savefig("seizure1_norm.png", dpi=400)



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

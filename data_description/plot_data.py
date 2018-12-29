import numpy as np
import matplotlib.pyplot as plt
import copy


""" Load datasets """
n_clip = 3
ieeg = {}
for c in range(1, n_clip+1):

    path = f"/home/phait/datasets/ieeg/TWH056_Day-504_Clip-0-{c}.npz"     # server
    # path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

    with np.load(path) as data:
        temp = dict(data)
        ieeg[c] = (temp['ieeg']).T


""" Variables """
electrodes = 15
timestamps = 10000
interval = {1: {'start': 1500000, 'end': 1800000},
            2: {'start': 50000, 'end': 350000},
            3: {'start': 0, 'end': 300000}}
seizure = {1: {'start': 1684381 - interval[1]['start'], 'end': 1699381 - interval[1]['start']},
           2: {'start': 188013 - interval[2]['start'], 'end': 201013 - interval[2]['start']},
           3: {'start': 96699 - interval[3]['start'], 'end': 110699 - interval[3]['start']}}

ieeg_norm = copy.deepcopy(ieeg)
value = 300


""" Creating normalized ieeg """
for c in range(1, n_clip+1):
    for i in range(0, 90):
        ieeg_norm[c][:, i] = ieeg_norm[c][:, i] - np.mean(ieeg_norm[c][:, i])

    for i in range(1, 90+1):
        ieeg_norm[c][:, i-1] = np.array(ieeg_norm[c][:, i-1]) + (i*value)


""" Plots """
for c in range(1, n_clip+1):

    plt.figure(figsize=(13.0, 8.0))
    plt.plot(ieeg[c][interval[c]['start']:interval[c]['end'], :],
             linewidth=0.4)
    plt.axvline(x=seizure[c]['start'])
    plt.axvline(x=seizure[c]['end'])
    plt.xlabel("Time")
    plt.ylabel("Signal values")
    plt.title(f"IEEG")
    plt.savefig(f"seizure{c}.png", dpi=400)

    plt.figure(figsize=(13.0, 8.0))
    plt.plot(ieeg_norm[c][interval[c]['start']:interval[c]['end'], :],
             linewidth=0.4)
    plt.yticks([])
    plt.axvline(x=seizure[c]['start'])
    plt.axvline(x=seizure[c]['end'])
    plt.xlabel("Time")
    plt.ylabel("Signal values")
    plt.title(f"IEEG")
    plt.savefig(f"seizure{c}_norm.png", dpi=400)


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
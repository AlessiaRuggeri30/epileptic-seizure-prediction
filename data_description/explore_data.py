'''
Dataset keys:
    ['ieeg', 'ieeg_mn', 'szr_bool', 'time_of_day_sec', 'srate_hz', 'day_since_implant']
The most interesting for me are:
    - ieeg:                 matrix with rows representing electrodes and columns representing timestamps;
    - time_of_day_sec:      timestamps in which the electrodes signal are measured;
    - szr_bool:             labels that indicates if in that timestamp there is an ongoing seizure or not.
'''


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

''' Check how many timestamps with ongoing seizure there are '''
unique, counts = np.unique(data['szr_bool'], return_counts=True)
d = dict(zip(unique, counts))

''' Check how many seizure there are ('s' is the resulting dictionary of seizures)'''
s = {}
k = 1
for i in range(1, data['time_of_day_sec'].shape[0]):
    if data['szr_bool'][i] == True and data['szr_bool'][i-1] == False:
        s[f"seizure{k}"] = {'start_idx': i-1}
    if data['szr_bool'][i] == False and data['szr_bool'][i-1] == True:
        s[f"seizure{k}"]['end_idx'] = i-1
        k += 1
print(s)

''' Check if it did miss some seizure in the middle '''
# print(np.where(data['szr_bool'][s['seizure1']['start_idx']:s['seizure1']['end_idx']] == False))

''' Check standard deviation on first timestamps, seizure timestamps and higher intensity timestamps'''
# print(np.std(data['ieeg'][0:15000]))
# print(np.std(data['ieeg'][:, s['seizure1']['start_idx']:s['seizure1']['end_idx']]))
# print(np.std(data['ieeg'][:, 1692500:s['seizure1']['end_idx']]))
''' Note tha standard deviation doesn't change inside and outside a seizure.
    This means that the seizure is not characterised by che intensity of the signal,
    but by the relations between the electrodes signal variations.'''


# TODO correlation matrix (heatmap)
# print(np.correlate(data['ieeg'][0:15000]))
# print(np.correlate(data['ieeg'][:, s['seizure1']['start_idx']:s['seizure1']['end_idx']]))
# print(np.correlate(data['ieeg'][:, 1692500:s['seizure1']['end_idx']]))


''' Write data description into file '''
file_name = "data_description.txt"
with open(file_name, 'w') as file:
    file.write("DATA DESCRIPTION\n\n")

    file.write("Dataset keys:\n")
    file.write(str(list(data.keys())) + "\n\n")

    file.write(f"Number of electrodes: {data['ieeg'].shape[0]}\n")
    file.write(f"Number of timestamps: {data['time_of_day_sec'].shape[0]}\n\n")

    file.write(f"Max voltage:\t{np.max(data['ieeg'])}\n")
    file.write(f"Min voltage:\t{np.min(data['ieeg'])}\n")
    file.write(f"Voltage mean:\t{np.mean(data['ieeg'])}\n")
    file.write(f"Voltage std:\t{np.std(data['ieeg'])}\n")

    file.write(f"Number of total timestamps with ongoing seizure:\t{d[True]}\n")
    file.write(f"Number of total timestamps without ongoing seizure:\t{d[False]}\n\n")

    file.write(f"Seizures:\n")
    file.write(str(s) + "\n\n")

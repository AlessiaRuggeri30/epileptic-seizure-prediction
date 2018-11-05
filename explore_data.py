import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

''' Load dataset '''
with np.load('dataset/TWH056_Day-504_Clip-0-1.npz') as data:
    data = dict(data)

'''
Dataset keys:
    ['ieeg', 'ieeg_mn', 'szr_bool', 'time_of_day_sec', 'srate_hz', 'day_since_implant']
The most interesting for me are:
    - ieeg:                 matrix with rows representing electrodes and columns representing timestamps;
    - time_of_day_sec:      timestamps in which the electrodes signal are measured;
    - szr_bool:             labels that indicates if in that timestamp there is an ongoing seizure or not.
'''


def rms(nparray):
    return np.sqrt(np.mean(nparray**2))


''' Check how many timestamps with ongoing seizure there are '''
unique, counts = np.unique(data['szr_bool'], return_counts=True)
d = dict(zip(unique, counts))

''' Check how many seizure there are '''
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
print(np.where(data['szr_bool'][s['seizure1']['start_idx']:s['seizure1']['end_idx']] == False))

print(rms(data['ieeg'][0:15000]))
print(rms(data['ieeg'][:, s['seizure1']['start_idx']:s['seizure1']['end_idx']]))





''' Write data description into file '''
with open('data_description.txt', 'w') as file:
    file.write("DATA DESCRIPTION\n\n")

    file.write("Dataset keys:\n")
    file.write(str(list(data.keys())) + "\n\n")

    file.write(f"Number of electrodes: {data['ieeg'].shape[0]}\n")
    file.write(f"Number of timestamps: {data['time_of_day_sec'].shape[0]}\n\n")

    file.write(f"Max voltage:\t{np.max(data['ieeg'])}\n")
    file.write(f"Min voltage:\t{np.min(data['ieeg'])}\n")
    file.write(f"Voltage mean:\t{np.mean(data['ieeg'])}\n")
    file.write(f"Voltage rms:\t{rms(data['ieeg'])}\n")

    file.write(f"Number of total timestamps with ongoing seizure:\t{d[True]}\n")
    file.write(f"Number of total timestamps without ongoing seizure:\t{d[False]}\n\n")

    file.write(f"Seizures:\n")
    file.write(str(s) + "\n\n")

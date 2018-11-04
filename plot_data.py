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

sns.countplot(data['szr_bool'])
plt.show()

ieeg = pd.DataFrame(data['ieeg'])
electrodes = 10
timestamps = 10000
ieeg_subset = ieeg.loc[0:(electrodes-1), 0:(timestamps-1)]
ieeg_subset = ieeg_subset.T

fig = sns.lineplot(data=ieeg_subset, hue='szr_bool', dashes=False, legend=False)
plt.xlabel("Time")
plt.ylabel("Signal values")
plt.title("EEG of first 60 electrodes for first 1000 timestamps")
plt.show()
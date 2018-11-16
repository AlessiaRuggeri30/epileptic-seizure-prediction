import numpy as np
import pandas as pd
import pyedflib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
sns.set()
sns.set_style('whitegrid')


path = "dataset/CHB-MIT Scalp EEG Database/chb18/chb18_29.edf"

''' Load dataset '''
data = pyedflib.EdfReader(path)

''' Create dataset matrix '''
n = data.signals_in_file
signal_labels = data.getSignalLabels()
sigbufs = np.zeros((n, data.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = data.readSignal(i)
nrow, ncol = sigbufs.shape

''' Convert np array in dataframe and transpose '''
ieeg = pd.DataFrame(sigbufs)
ieeg = ieeg.T

''' Normalize matrix values to have a nice plot with equidistant signals'''
value = 150
for i in range(1, nrow+1):
    ieeg[i-1] = np.array(ieeg[i-1]) + (i*value)

electrodes = 28
seizure_start = 0
seizure_end = 5000


''' Plot of first electrodes during seizure timestamps'''
ieeg_subset = ieeg.loc[seizure_start:seizure_end, 0:(electrodes-1)]

fig2 = sns.lineplot(data=ieeg_subset, hue='szr_bool', dashes=False, legend=False)
plt.xlabel("Time")
plt.ylabel("Signal values")
plt.title(f"EEG of first {electrodes} electrodes during seizure timestamps")
plt.yticks([])
plt.axvline(x=3477)
plt.axvline(x=3527)
plt.show()

''' Correlation heatmaps of initial timestamps'''
corr1 = ieeg[0:50].corr()
sns.heatmap(corr1, cmap="RdBu_r", center=0)
plt.title("First 100")
plt.show()

''' Correlation heatmaps of seizure timestamps'''
corr2 = ieeg[3477:3527].corr()
sns.heatmap(corr2, cmap="RdBu_r", center=0)
plt.title("First 100 of seizure")
plt.show()


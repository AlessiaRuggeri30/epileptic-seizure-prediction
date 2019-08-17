import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from utils.load_data import load_data

np.random.seed(42)

""" Global parameters """
statistics = True
seizures_plots = False
hist_plot = False
heatmap = False

""" Import dataset """
X, y, dataset, seizure = load_data(local=True)
x1, x2, x3 = (np.transpose(X[1]), np.transpose(X[2]), np.transpose(X[3]))
y1, y2, y3 = (y[1], y[2], y[3])
dataset = np.transpose(dataset)
dataset_y = np.concatenate((y1, y2, y3))

""" First info """
n_seizures_1, n_seizures_2, n_seizures_3 = (len(y1[y1==1]), len(y2[y2==1]), len(y3[y3==1]))
tot_seizures = sum([n_seizures_1, n_seizures_2, n_seizures_3])

if statistics:
    print(f"Num electrodes: {dataset.shape[0]}")
    print(f"Num time steps: {dataset.shape[1]}")
    print()

    print(f"Num positive time steps in three seizures: {n_seizures_1}, {n_seizures_2}, {n_seizures_3}")
    print(f"Total num positive time steps: {tot_seizures}")
    print()

    """ Basic statistical measurements """
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    range_value = max_value - min_value
    print(f"Max, min, range: {max_value}, {min_value}, {range_value}")

    ranges_electodes = set()
    std_electodes = set()
    min_range = 100000
    max_range = 0
    for electrode in dataset:
        max_value = np.max(electrode)
        min_value = np.min(electrode)
        range_value = max_value - min_value
        if range_value < min_range: min_range = range_value
        if range_value > max_range: max_range = range_value
        std_value = np.std(electrode)
        ranges_electodes.add(range_value)
        std_electodes.add(std_value)

    ranges_electodes_mean = np.mean(list(ranges_electodes))
    std_electodes_mean = np.mean(list(std_electodes))
    print(f"Average range of single electrode values: {max_value}, min {min_range}, max {max_range}")
    print(f"Average std of single electrode values: {std_electodes_mean}")
    print()

    t_nonseizure = dataset[:,dataset_y==0]
    t_seizure = dataset[:, dataset_y == 1]
    std_nonseizure = np.mean(np.std(t_nonseizure, axis=1))
    std_seizure = np.mean(np.std(t_seizure, axis=1))
    print(f"Average std of single electrode values during nonseizure: {std_nonseizure}")
    print(f"Average std of single electrode values during seizure: {std_seizure}")
    print()

""" Plots """
if seizures_plots:
    for c in range(1, 4):
        plt.figure(figsize=(13.0, 8.0))
        plt.plot(X[c], linewidth=0.4)
        plt.axvline(x=seizure[c]['start'], color='r')
        plt.axvline(x=seizure[c]['end'], color='r')
        plt.xticks(np.arange(0, 150000+1, 10000))
        plt.yticks(np.arange(-10000, 10000+1, 2000))
        plt.xlabel("Time steps")
        plt.ylabel("Signal voltage")
        plt.tight_layout()
        plt.savefig(f"./plots/plot_seizure{c}.png", dpi=400)

if hist_plot:
    plt.figure()
    plt.bar([0, 1], height= [(dataset.shape[1]-tot_seizures), tot_seizures])
    plt.xticks([0, 1], ['negative class', 'positive class'])
    plt.yticks(np.arange(0, 450000+1, 50000))
    plt.grid(linestyle='dotted')
    plt.xlabel("Classes")
    plt.ylabel("Number of time steps")
    plt.tight_layout()
    plt.savefig(f"./plots/hist_classes.png")

if heatmap:
    cmap = "Blues"
    nonseizure_slice = x1[:,50000:52500]
    # nonseizure_slice = x2[:,50000:52500]
    # nonseizure_slice = x3[:,40000:42500]
    corr1 = np.corrcoef(nonseizure_slice)
    plt.figure()
    heatmap = plt.pcolor(corr1, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(heatmap, ticks=np.arange(-1, 1.1, 0.2))
    plt.xlabel("Electrodes")
    plt.ylabel("Electrodes")
    plt.tight_layout()
    plt.savefig(f"./plots/corr_nonseizure.png")

    seizure_slice = x1[:,105000:107500]
    # seizure_slice = x2[:,90000:92500]
    # seizure_slice = x2[:,90000:92500]
    corr2 = np.corrcoef(seizure_slice)
    plt.figure()
    heatmap = plt.pcolor(corr2, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(heatmap, ticks=np.arange(-1, 1.1, 0.2))
    plt.xlabel("Electrodes")
    plt.ylabel("Electrodes")
    plt.tight_layout()
    plt.savefig(f"./plots/corr_seizure.png")


print()


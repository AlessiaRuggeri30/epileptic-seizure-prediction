import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


""" Variables """
n_clip = 3
X = {}
y = {}
interval = {1: {'start': 1640000, 'end': 1740000},
            2: {'start': 150000, 'end': 250000},
            3: {'start': 60000, 'end': 160000}}
seizure = {1: {'start': 1684381 - interval[1]['start'], 'end': 1699381 - interval[1]['start']},
           2: {'start': 188013 - interval[2]['start'], 'end': 201013 - interval[2]['start']},
           3: {'start': 96699 - interval[3]['start'], 'end': 110699 - interval[3]['start']}}


""" Load datasets """

for c in range(1, n_clip + 1):
    path = f"/home/phait/datasets/ieeg/TWH056_Day-504_Clip-0-{c}.npz"  # server
    # path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

    with np.load(path) as data:
        data = dict(data)

    data['szr_bool'] = data['szr_bool'].astype(int)  # one-hot encoding

    X[c] = data['ieeg'].T[interval[c]['start']:interval[c]['end']]
    y[c] = data['szr_bool'][interval[c]['start']:interval[c]['end']]


""" Select training set and test set """
X_training = np.concatenate((X[2], X[3]), axis=0)
y_training = np.concatenate((y[2], y[3]), axis=0)
X_test = X[1]
y_test = y[1]

print(X_training.shape, y_training.shape)
print(X_test.shape, y_test.shape)

print("Creating the random forest classifier...")
n_estimators = 100
max_depth = 10
random_state = 0
on_seizure = False
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                             random_state=random_state, n_jobs=-1, class_weight="balanced")

print("Fitting training data to the random forest classifier...")
clf.fit(X_training, y_training)

print("Predicting values on test data...")
predictions = clf.predict(X_test)
errors = abs(predictions - y_test)

print("Results")
loss = round(brier_score_loss(y_test, predictions), 4)
accuracy = round(accuracy_score(y_test, predictions), 4)
roc_auc_score = round(roc_auc_score(y_test, predictions), 4)
print(f"\tLoss:\t\t{loss}")
print(f"\tAccuracy:\t{accuracy}")
print(f"\tRoc:\t\t{roc_auc_score}")


""" Write results into file """
if on_seizure:
    dataset = "on_seizure_data"
else:
    dataset = "on_whole_data"

file_name = f"random_forest(n_estimators={n_estimators},max_depth={max_depth}," \
    f"random_state={random_state}, balanced)-{dataset}.txt"
with open(file_name, 'w') as file:
    file.write("EXPERIMENT: RANDOM FOREST\n\n")

    file.write("Parameters\n")
    file.write(f"\tn_estimators:\t{n_estimators}\n")
    file.write(f"\tmax_depth:\t\t{max_depth}\n")
    file.write(f"\trandom_state:\t{random_state}\n")
    file.write(f"\tbalanced\n")
    file.write(f"\tdataset:\t\t{dataset}\n\n")

    file.write("Data shape\n")
    file.write(f"\tX_training shape:\t{X_training.shape}\n")
    file.write(f"\ty_training shape:\t{y_training.shape}\n")
    file.write(f"\tX_test shape:\t\t{X_test.shape}\n")
    file.write(f"\ty_test shape:\t\t{y_test.shape}\n\n")

    file.write("Results\n")
    file.write(f"\tLoss:\t\t{loss}\n")
    file.write(f"\tAccuracy:\t{accuracy}\n")
    file.write(f"\tRoc:\t{roc_auc_score}\n\n")


""" Plots """
plt.subplot(2, 1, 1)
plt.plot(y_test)
plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
plt.subplot(2, 1, 2)
plt.plot(predictions)
plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
plt.savefig("./plots/predictions.png")
# plt.savefig("./plots/predictions_nonbalanced.png")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, accuracy_score

''' Load dataset '''
path = "/home/phait/datasets/ieeg/TWH056_Day-504_Clip-0-1.npz"     # server
# path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

with np.load(path) as data:
    data = dict(data)

data['szr_bool'] = data['szr_bool'].astype(int)     # one-hot encoding

X = data['ieeg'].T  # [1670000:]
y = data['szr_bool']    # [1670000:]
print(X.shape, y.shape)

print("Creating the random forest classifier...")
n_estimators = 100
max_depth = 10
random_state = 0
on_seizure = False
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                             random_state=random_state)

print("Fitting training data to the random forest classifier...")
clf.fit(X, y)

print("Predicting values on test data...")
predictions = clf.predict(X)
errors = abs(predictions - y)

print("Results")
loss = round(brier_score_loss(y, predictions), 4)
accuracy = round(accuracy_score(y, predictions), 4)
print(f"\tLoss:\t\t{loss}")
print(f"\tAccuracy:\t{accuracy}")


''' Write results into file '''
if on_seizure:
    dataset = "on_seizure_data"
else:
    dataset = "on_whole_data"

file_name = f"random_forest(n_estimators={n_estimators},max_depth={max_depth}," \
    f"random_state={random_state})-{dataset}.txt"
with open(file_name, 'w') as file:
    file.write("EXPERIMENT: RANDOM FOREST\n\n")

    file.write("Parameters\n")
    file.write(f"\tn_estimators:\t{n_estimators}\n")
    file.write(f"\tmax_depth:\t\t{max_depth}\n")
    file.write(f"\trandom_state:\t{random_state}\n")
    file.write(f"\tdataset:\t\t{dataset}\n\n")

    file.write("Data shape\n")
    file.write(f"\tX shape:\t{X.shape}\n")
    file.write(f"\ty shape:\t{y.shape}\n\n")

    file.write("Results\n")
    file.write(f"\tLoss:\t\t{loss}\n")
    file.write(f"\tAccuracy:\t{accuracy}\n\n")


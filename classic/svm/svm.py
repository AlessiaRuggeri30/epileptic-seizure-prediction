import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import brier_score_loss, accuracy_score

n_clip = 3
X = {}
y = {}
delimiters = {1: {'start': 1640000, 'end': 1740000},
              2: {'start': 150000, 'end': 250000},
              3: {'start': 60000, 'end': 160000}}

for c in range(1, n_clip+1):

    ''' Load dataset '''
    path = f"/home/phait/datasets/ieeg/TWH056_Day-504_Clip-0-{c}.npz"     # server
    # path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

    with np.load(path) as data:
        data = dict(data)

    data['szr_bool'] = data['szr_bool'].astype(int)     # one-hot encoding

    X[c] = data['ieeg'].T[delimiters[c]['start']:delimiters[c]['end']]
    y[c] = data['szr_bool'][delimiters[c]['start']:delimiters[c]['end']]


''' Select training set and test set '''
X_training = np.concatenate((X[2], X[3]), axis=0)
y_training = np.concatenate((y[2], y[3]), axis=0)
X_test = X[1]
y_test = y[1]

print(X_training.shape, y_training.shape)
print(X_test.shape, y_test.shape)

print("Creating the svm classifier...")
gamma = 'scale'
weighted = False
on_seizure = True
clf = svm.SVC(gamma=gamma)

print("Fitting training data to the svm classifier...")
if weighted is False:
    clf.fit(X_training, y_training)
else:
    weighted = class_weight = {1: 10}
    clf.fit(X_training, y_training, class_weight=class_weight)

print("Predicting values on test data...")
predictions = clf.predict(X_test)
errors = abs(predictions - y_test)

print("Results")
loss = round(brier_score_loss(y_test, predictions), 4)
accuracy = round(accuracy_score(y_test, predictions), 4)
print(f"\tLoss:\t\t{loss}")
print(f"\tAccuracy:\t{accuracy}")


''' Write results into file '''
if on_seizure:
    dataset = "on_seizure_data"
else:
    dataset = "on_whole_data"

file_name = f"svm(gamma={gamma},weighted={weighted})" \
    f"-{dataset}.txt"
with open(file_name, 'w') as file:
    file.write("EXPERIMENT: SVM\n\n")

    file.write("Parameters\n")
    file.write(f"\tgamma:\t\t{gamma}\n")
    file.write(f"\tweighted:\t{weighted}\n")
    file.write(f"\tdataset:\t{dataset}\n\n")

    file.write("Data shape\n")
    file.write(f"\tX shape:\t{X.shape}\n")
    file.write(f"\ty shape:\t{y.shape}\n\n")

    file.write("Results\n")
    file.write(f"\tLoss:\t\t{loss}\n")
    file.write(f"\tAccuracy:\t{accuracy}\n\n")

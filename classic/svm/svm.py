import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import brier_score_loss, accuracy_score

''' Load dataset '''
# path = "../.spektral/datasets/ieeg/TWH056_Day-504_Clip-0-1.npz"     # server
path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

with np.load(path) as data:
    data = dict(data)

data['szr_bool'] = data['szr_bool'].astype(int)     # one-hot encoding

X = data['ieeg'].T[1670000:]
y = data['szr_bool'][1670000:]
print(X.shape, y.shape)

print("Creating the svm classifier...")
gamma = 'scale'
weighted = False
on_seizure = True
clf = svm.SVC(gamma=gamma)

print("Fitting training data to the svm classifier...")
if weighted is False:
    clf.fit(X, y)
else:
    weighted = class_weight = {1: 10}
    clf.fit(X, y, class_weight=class_weight)

print("Predicting values on test data...")
predictions = clf.predict(X)
errors = abs(predictions - y)

print("Results")
loss = round(brier_score_loss(y, predictions), 4)
accuracy = round(accuracy_score(y, predictions), 4)
print(f"\tLoss:\t\t{loss}")
print(f"\tAccuracy:\t{accuracy}")

# print(errors[14300:30000])


''' Write results into file '''
if on_seizure:
    dataset = "on_seizure_data"
else:
    dataset = "on_whole_data"

file_name = f"svm(gamma={gamma},weighted={weighted}," \
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

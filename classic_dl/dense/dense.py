import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Reshape, Flatten
from keras.regularizers import l2
from keras.preprocessing.sequence import TimeseriesGenerator
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

    if c == 1:
        dataset = X[c]
    else:
        dataset = np.concatenate((dataset, X[c]), axis=0)


# X[c]      samples from clip c
# y[c]      targets from clip c
# dataset   concatenation of samples from all clips (concatenation of all X[c], used for scaling on entire dataset)


# -----------------------------------------------------------------------------
# DATA PREPROCESSING
# -----------------------------------------------------------------------------


""" Select training set and test set """
X_train = np.concatenate((X[2], X[3]), axis=0)
y_train = np.concatenate((y[2], y[3]), axis=0)
X_test = X[1]
y_test = y[1]

n_positive = np.sum(y_train)
n_negative = len(y_train) - n_positive

""" Normalize data """
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler.fit(dataset)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

""" Reshape data """

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# -----------------------------------------------------------------------------
# MODEL BUILDING, TRAINING AND TESTING
# -----------------------------------------------------------------------------


""" Build the model """
num = 3

epochs = 20
batch_size = 32
units = 512
reg = l2(5e-4)
activation = 'tanh'
class_weight = {0: (len(y_train)/n_negative), 1: (len(y_train)/n_positive)}

model = Sequential()
model.add(Dense(units, activation=activation, kernel_regularizer=reg, batch_input_shape=(batch_size, 90)))
model.add(Dropout(0.5))
model.add(Dense(units, activation=activation, kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(256, activation=activation, kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

""" Fit the model """
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weight)

""" Save and reload the model """
model.save(f"dense_model{num}.h5")
del model
model = load_model(f"dense_model{num}.h5")
# model = load_model(f"dense_model3.h5")


# -----------------------------------------------------------------------------
# RESULTS EVALUATION
# -----------------------------------------------------------------------------


""" Predictions on training data """
print("Predicting values on training data...")
predictions_train = model.predict(X_train, batch_size=batch_size)
predictions_train[predictions_train <= 0.5] = 0
predictions_train[predictions_train > 0.5] = 1

print("Results on training data")
loss_train = round(log_loss(y_train, predictions_train, eps=1e-7), 4)  # for the clip part, eps=1e-15 is too small for float32
accuracy_train = round(accuracy_score(y_train, predictions_train), 4)
roc_auc_score_train = round(roc_auc_score(y_train, predictions_train), 4)
print(f"\tLoss:\t\t{loss_train}")
print(f"\tAccuracy:\t{accuracy_train}")
print(f"\tRoc:\t\t{roc_auc_score_train}")

""" Predictions on test data """
loss_keras, metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
loss_keras = round(loss_keras, 4)
print(f"\tLoss:\t\t{loss_keras}")
print(f"\tAccuracy:\t{metrics}")

print("Predicting values on test data...")
predictions = model.predict(X_test, batch_size=batch_size)
sigmoid = np.copy(predictions)
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

print("Results on test data")
loss = round(log_loss(y_test, predictions, eps=1e-7), 4)  # for the clip part, eps=1e-15 is too small for float32
accuracy = round(accuracy_score(y_test, predictions), 4)
roc_auc_score = round(roc_auc_score(y_test, predictions), 4)
print(f"\tLoss:\t\t{loss}")
print(f"\tAccuracy:\t{accuracy}")
print(f"\tRoc:\t\t{roc_auc_score}")


# -----------------------------------------------------------------------------
# EXPERIMENT RESULTS SUMMARY
# -----------------------------------------------------------------------------


exp = "exp" + str(num)
file_name = exp + "_dense.txt"
string_list = []
model.summary(print_fn=lambda x: string_list.append(x))
summary = "\n".join(string_list)

with open(file_name, 'w') as file:
    file.write(f"EXPERIMENT {num}: DENSE NEURAL NETWORK\n\n")

    file.write("Parameters\n")
    file.write(f"\tepochs:\t\t\t{epochs}\n")
    file.write(f"\tbatch_size:\t\t{batch_size}\n")
    file.write(f"\treg:\t\t\tl2(5e-4)\n")
    file.write(f"\tactivation:\t\t{activation}\n")
    file.write(f"\tclass_weight:\t{str(class_weight)}\n\n")

    file.write("Model\n")
    file.write(f"{summary}\n\n")

    file.write("Data shape\n")
    file.write(f"\tX_train shape:\t{X_train.shape}\n")
    file.write(f"\ty_train shape:\t{y_train.shape}\n")
    file.write(f"\tX_test shape:\t{X_test.shape}\n")
    file.write(f"\ty_test shape:\t{y_test.shape}\n\n")

    file.write("Results on train set\n")
    file.write(f"\tLoss:\t\t{loss_train}\n")
    file.write(f"\tAccuracy:\t{accuracy_train}\n")
    file.write(f"\tRoc_auc:\t{roc_auc_score_train}\n\n")

    file.write("Results on test set\n")
    file.write(f"\tLoss_keras:\t{loss_keras}\n")
    file.write(f"\tLoss:\t\t{loss}\n")
    file.write(f"\tAccuracy:\t{accuracy}\n")
    file.write(f"\tRoc_auc:\t{roc_auc_score}\n")


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------


plt.subplot(2, 1, 1)
plt.plot(y_train)
plt.subplot(2, 1, 2)
plt.plot(predictions_train)
plt.savefig(f"./plots/{exp}-predictions_train.png")
plt.close()

plt.subplot(2, 1, 1)
plt.plot(y_test)
plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
plt.subplot(2, 1, 2)
plt.plot(predictions)
plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
plt.savefig(f"./plots/{exp}-predictions.png")
plt.close()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


plt.figure(figsize=(15.0, 8.0))
plt.plot(sigmoid)
plt.plot(running_mean(sigmoid, 1000))
plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
plt.savefig(f"./plots/{exp}-sigmoid.png", dpi=400)


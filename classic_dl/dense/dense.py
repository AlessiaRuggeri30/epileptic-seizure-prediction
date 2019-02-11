import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, InputLayer, Reshape, Flatten
from keras.regularizers import l2
from keras.preprocessing.sequence import TimeseriesGenerator

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
epochs = 30
batch_size = 32
units = 128
reg = l2(5e-4)
class_weight = {0: (len(y_train)/n_negative), 1: (len(y_train)/n_positive)}

model = Sequential()
model.add(Dense(units, activation='tanh', kernel_regularizer=reg, batch_input_shape=(batch_size, 90)))
model.add(Dropout(0.5))
model.add(Dense(units, activation='tanh', kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(units, activation='tanh', kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

""" Fit the model """
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weight)

""" Save and reload the model """
model.save('dense_model.h5')
del model
model = load_model('dense_model.h5')


# -----------------------------------------------------------------------------
# RESULTS EVALUATION
# -----------------------------------------------------------------------------


""" Predictions on training data """
print("Predicting values on training data...")
predictions_train = model.predict(X_train, batch_size=batch_size)
# predictions_train = predictions_train.reshape(-1)
predictions_train[predictions_train <= 0.5] = 0
predictions_train[predictions_train > 0.5] = 1

print("Results on training data")
loss_train = round(log_loss(y_train, predictions_train, eps=1e-7), 4)  # for the clip part, eps=1e-15 is too small for float32
accuracy_train = round(accuracy_score(y_train, predictions_train), 4)
roc_auc_score_train = round(roc_auc_score(y_train, predictions_train), 4)
print(f"\tLoss:\t\t{loss_train}")
print(f"\tLoss_norm:\t\t{loss_train/batch_size}")
print(f"\tAccuracy:\t{accuracy_train}")
print(f"\tRoc:\t\t{roc_auc_score_train}")

""" Predictions on test data """
loss, metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Loss: {loss}")
print(f"Accuracy: {metrics}")

print("Predicting values on test data...")
predictions = model.predict(X_test, batch_size=batch_size)
# predictions = predictions.reshape(-1)
sigmoid = np.copy(predictions)
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

print("Results on test data")
loss = round(log_loss(y_test, predictions, eps=1e-7), 4)  # for the clip part, eps=1e-15 is too small for float32
accuracy = round(accuracy_score(y_test, predictions), 4)
roc_auc_score = round(roc_auc_score(y_test, predictions), 4)
print(f"\tLoss:\t\t{loss}")
print(f"\tLoss_norm:\t\t{loss/batch_size}")
print(f"\tAccuracy:\t{accuracy}")
print(f"\tRoc:\t\t{roc_auc_score}")


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

#
# plt.subplot(2, 1, 1)
# plt.plot(y_train)
# plt.subplot(2, 1, 2)
# plt.plot(predictions_train)
# plt.savefig("./plots/predictions_train.png")
# plt.close()
#
# plt.subplot(2, 1, 1)
# plt.plot(y_test)
# plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
# plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
# plt.subplot(2, 1, 2)
# plt.plot(predictions)
# plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
# plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
# plt.savefig("./plots/predictions.png")
# plt.close()
#
#
# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / float(N)
#
#
# plt.figure(figsize=(15.0, 8.0))
# plt.plot(sigmoid)
# plt.plot(running_mean(sigmoid, 1000))
# plt.axvline(x=seizure[1]['start'], color="orange", linewidth=0.5)
# plt.axvline(x=seizure[1]['end'], color="orange", linewidth=0.5)
# plt.savefig("./plots/sigmoid.png", dpi=400)


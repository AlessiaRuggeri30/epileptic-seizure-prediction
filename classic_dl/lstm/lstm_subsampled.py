import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.regularizers import l2
from sklearn import preprocessing
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from keras import callbacks
from utils import generate_indices

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
look_back = 100
stride = 1  # Keep this =1 so that you keep all positive samples
predicted_timestamps = 1
shift = -1  # starting from the position len(sequence) + 1

""" Generate sequences """
# Generate sequences by computing indices
inputs_indices_seq, target_indices_seq =  \
    generate_indices([y_train],             # Targets associated to X_train (same shape[0])
                     look_back,             # Length of input sequences
                     stride=stride,         # Stride between windows
                     target_steps_ahead=0,  # How many steps ahead to predict (x[t], ..., x[t+T] -> y[t+T+k])
                     subsample=True,        # Whether to subsample
                     subsampling_factor=2   # Keep this many negative samples w.r.t. to positive ones
                     )
X_train = X_train[inputs_indices_seq]
y_train = y_train[target_indices_seq]

# -----------------------------------------------------------------------------
# MODEL BUILDING, TRAINING AND TESTING
# -----------------------------------------------------------------------------
""" Build the model """
num = 2
epochs = 10
batch_size = 64
units = 128
reg = l2(5e-4)
activation = 'tanh'
class_weight = {0: (len(y_train) / n_negative), 1: (len(y_train) / n_positive)}

model = Sequential()
model.add(LSTM(units, activation=activation, kernel_regularizer=reg, input_shape=(look_back, 90), return_sequences=True))
model.add(LSTM(units, activation=activation, kernel_regularizer=reg))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

""" Fit the model """
callbacks = [
    callbacks.TensorBoard('plots'),
]
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          class_weight=class_weight,
          callbacks=callbacks)

""" Save and reload the model """
model.save(f"lstm_model{num}.h5")
# del model
# model = load_model(f"lstm_model{num}.h5")
# model = load_model(f"lstm_model1.h5")

# -----------------------------------------------------------------------------
# RESULTS EVALUATION
# -----------------------------------------------------------------------------
""" Predictions on training data """
print("Predicting values on training data...")
predictions_train = model.predict(X_train, batch_size=batch_size).flatten()

print("Results on training data")
loss_train = log_loss(y_train, predictions_train, eps=1e-7)  # for the clip part, eps=1e-15 is too small for float32
accuracy_train = accuracy_score(y_train, np.round(predictions_train))
roc_auc_score_train = roc_auc_score(y_train, predictions_train)
print(f"\tLoss:    \t{loss_train:.4f}")
print(f"\tAccuracy:\t{accuracy_train:.4f}")
print(f"\tROC-AUC: \t{roc_auc_score_train:.4f}")

""" Predictions on test data """
loss_keras, metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print(f"\tLoss:    \t{loss_keras:.4f}")
print(f"\tAccuracy:\t{metrics:.4f}")

print("Predicting values on test data...")
predictions_test = model.predict(X_test, batch_size=batch_size).flatten()

print("Results on test data")
loss_test = log_loss(y_test, predictions_test, eps=1e-7)  # for the clip part, eps=1e-15 is too small for float32
accuracy_test = accuracy_score(y_test, np.round(predictions_test))
roc_auc_score_test = roc_auc_score(y_test, predictions_test)
print(f"\tLoss:    \t{loss_test:.4f}")
print(f"\tAccuracy:\t{accuracy_test:.4f}")
print(f"\tROC-AUC: \t{roc_auc_score_test:.4f}")


# -----------------------------------------------------------------------------
# EXPERIMENT RESULTS SUMMARY
# -----------------------------------------------------------------------------
exp = "exp" + str(num)
file_name = exp + "_lstm.txt"
string_list = []
model.summary(print_fn=lambda x: string_list.append(x))
summary = "\n".join(string_list)

with open(file_name, 'w') as file:
    file.write(f"EXPERIMENT {num}: LSTM NEURAL NETWORK\n\n")
    file.write("Parameters\n")
    file.write(f"\tepochs:\t\t\t\t\t{epochs}\n")
    file.write(f"\tbatch_size:\t\t\t\t{batch_size}\n")
    file.write(f"\treg:\t\t\t\t\tl2(5e-4)\n")
    file.write(f"\tactivation:\t\t\t\t{activation}\n")
    file.write(f"\tclass_weight:\t\t\t{str(class_weight)}\n")
    file.write(f"\tlook_back:\t\t\t\t{look_back}\n")
    file.write(f"\tstride:\t\t\t\t\t{stride}\n")
    file.write(f"\tpredicted_timestamps:\t{predicted_timestamps}\n")
    file.write(f"\tshift:\t\t\t\t\t{shift}\n\n")

    file.write("Model\n")
    file.write(f"{summary}\n\n")

    file.write("Data shape\n")
    file.write(f"\tX_train shape:\t{X_train.shape}\n")
    file.write(f"\ty_train shape:\t{y_train.shape}\n")
    file.write(f"\tX_test shape: \t{X_test.shape}\n")
    file.write(f"\ty_test shape: \t{y_test.shape}\n")
    file.write(f"\tNumber of train samples:\t{len(y_train)}\n")
    file.write(f"\tNumber of test samples: \t{len(y_test)}\n\n")

    file.write("Results on train set\n")
    file.write(f"\tLoss:\t\t{loss_train}\n")
    file.write(f"\tAccuracy:\t{accuracy_train}\n")
    file.write(f"\tRoc_auc:\t{roc_auc_score_train}\n\n")

    file.write("Results on test set\n")
    file.write(f"\tLoss_keras:\t{loss_keras}\n")
    file.write(f"\tLoss:\t\t{loss_test}\n")
    file.write(f"\tAccuracy:\t{accuracy_test}\n")
    file.write(f"\tRoc_auc:\t{roc_auc_score_test}\n")


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
plt.subplot(2, 1, 2)
plt.plot(predictions_test)
plt.savefig(f"./plots/{exp}-predictions.png")
plt.close()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# plt.figure(figsize=(15.0, 8.0))
# plt.plot(sigmoid)
# plt.plot(running_mean(sigmoid, 1000))
# plt.savefig(f"./plots/{exp}-sigmoid.png", dpi=400)

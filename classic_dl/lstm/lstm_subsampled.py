import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.models import Sequential, load_model
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from keras import callbacks
from lstm_model import build_lstm_model
import sys
sys.path.append("....")
from utils.utils import add_experiment, save_experiments, generate_indices
from utils.load_data import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

X, y, dataset, seizure = load_data()

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
scaler = StandardScaler()
scaler.fit(dataset)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

""" Generate sequences """
look_back = 100
stride = 1  # Keep this =1 so that you keep all positive samples
predicted_timestamps = 1
target_steps_ahead = 0  # starting from the position len(sequence)
subsampling_factor = 2

# Generate sequences by computing indices for training data
inputs_indices_seq, target_indices_seq =  \
    generate_indices([y_train],                              # Targets associated to X_train (same shape[0])
                     look_back,                              # Length of input sequences
                     stride=stride,                          # Stride between windows
                     target_steps_ahead=target_steps_ahead,  # How many steps ahead to predict (x[t], ..., x[t+T] -> y[t+T+k])
                     subsample=True,                         # Whether to subsample
                     subsampling_factor=subsampling_factor   # Keep this many negative samples w.r.t. to positive ones
                     )
X_train = X_train[inputs_indices_seq]
y_train = y_train[target_indices_seq]

# Generate sequences by computing indices for test data
inputs_indices_seq, target_indices_seq =  \
    generate_indices([y_test],                              # Targets associated to X_train (same shape[0])
                     look_back,                              # Length of input sequences
                     stride=stride,                          # Stride between windows
                     target_steps_ahead=target_steps_ahead,  # How many steps ahead to predict (x[t], ..., x[t+T] -> y[t+T+k])
                     )
X_test = X_test[inputs_indices_seq]
y_test = y_test[target_indices_seq]

""" Shuffle training data """
X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# -----------------------------------------------------------------------------
# MODEL BUILDING, TRAINING AND TESTING
# -----------------------------------------------------------------------------
""" Build the model """
num = 3
exp = "exp" + str(num)
file_name = exp + "_lstm.txt"

epochs = 10
batch_size = 64
depth_lstm = 2
depth_dense = 1
units_lstm = 128
reg = l2(5e-4)
activation = 'tanh'
batch_norm = True
dropout = 0.5
class_weight = {0: (len(y_train) / n_negative), 1: (len(y_train) / n_positive)}

model = build_lstm_model(depth_lstm, depth_dense, units_lstm, reg, activation, batch_norm, dropout)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

""" Fit the model """
callbacks = [
    callbacks.TensorBoard(log_dir=f".logs/{exp}"),
]
model.fit(X_train_shuffled, y_train_shuffled,
          batch_size=batch_size,
          epochs=epochs,
          class_weight=class_weight,
          callbacks=callbacks)

""" Save and reload the model """
model.save(f"models/lstm_model{num}.h5")
del model
model = load_model(f"models/lstm_model{num}.h5")

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
string_list = []
model.summary(print_fn=lambda x: string_list.append(x))
summary = "\n".join(string_list)

with open(f"results/{file_name}", 'w') as file:
    file.write(f"EXPERIMENT {num}: LSTM NEURAL NETWORK\n\n")

    file.write("Parameters\n")
    file.write(f"\tepochs:\t\t\t{epochs}\n")
    file.write(f"\tbatch_size:\t\t{batch_size}\n")
    file.write(f"\tdepth_lstm:\t\t{depth_lstm}\n")
    file.write(f"\tdepth_dense:\t{depth_dense}\n")
    file.write(f"\tunits_lstm:\t\t{units_lstm}\n")
    file.write(f"\treg:\t\t\tl2(5e-4)\n")
    file.write(f"\tactivation:\t\t{activation}\n")
    file.write(f"\tbatch_norm:\t\t{str(batch_norm)}\n")
    file.write(f"\tdropout:\t\t{dropout}\n")
    file.write(f"\tclass_weight:\t{str(class_weight)}\n")
    file.write(f"\tlook_back:\t\t{look_back}\n")
    file.write(f"\tstride:\t\t\t{stride}\n")
    file.write(f"\tpredicted_timestamps:\t{predicted_timestamps}\n")
    file.write(f"\ttarget_steps_ahead:\t\t{target_steps_ahead}\n")
    file.write(f"\tsubsampling_factor:\t\t{subsampling_factor}\n\n")

    file.write("Model\n")
    file.write(f"{summary}\n\n")

    file.write("Data shape\n")
    file.write(f"\tX_train shape:\t{X_train.shape}\n")
    file.write(f"\ty_train shape:\t{y_train.shape}\n")
    file.write(f"\tX_test shape: \t{X_test.shape}\n")
    file.write(f"\ty_test shape: \t{y_test.shape}\n\n")

    file.write("Results on train set\n")
    file.write(f"\tLoss:\t\t{loss_train}\n")
    file.write(f"\tAccuracy:\t{accuracy_train}\n")
    file.write(f"\tRoc_auc:\t{roc_auc_score_train}\n\n")

    file.write("Results on test set\n")
    file.write(f"\tLoss_keras:\t{loss_keras}\n")
    file.write(f"\tLoss:\t\t{loss_test}\n")
    file.write(f"\tAccuracy:\t{accuracy_test}\n")
    file.write(f"\tRoc_auc:\t{roc_auc_score_test}\n")

experiments = "experiments_lstm"
hyperpar = ['', 'epochs', 'depth_lstm', 'depth_dense', 'units_lstm', 'activation',
            'batch_norm', 'dropout', 'look_back', 'target_steps_ahead',
            'subsampling_factor', 'loss', 'acc', 'roc-auc']
exp_hyperpar = [epochs, depth_lstm, depth_dense, units_lstm, activation,
                batch_norm, dropout, look_back, target_steps_ahead,
                subsampling_factor, loss_test, accuracy_test, roc_auc_score_test]
df = add_experiment(num, exp_hyperpar, experiments, hyperpar)
save_experiments(df, experiments)

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
# predictions_train[predictions_train <= 0.5] = 0
# predictions_train[predictions_train > 0.5] = 1
# sigmoid = np.copy(predictions_test)
# predictions_test[predictions_test <= 0.5] = 0
# predictions_test[predictions_test > 0.5] = 1

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

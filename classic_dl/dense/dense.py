import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from keras import callbacks
import sys
sys.path.append("....")
from utils.utils import add_experiment, save_experiments
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

""" Shuffle training data """
X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# -----------------------------------------------------------------------------
# MODEL BUILDING, TRAINING AND TESTING
# -----------------------------------------------------------------------------
""" Build the model """
num = 12
exp = "exp" + str(num)
file_name = exp + "_dense.txt"

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
callbacks = [
    callbacks.TensorBoard(log_dir=f".logs/{exp}"),
]
model.fit(X_train_shuffled, y_train_shuffled,
          batch_size=batch_size,
          epochs=epochs,
          class_weight=class_weight,
          callbacks=callbacks)

""" Save and reload the model """
model.save(f"models/dense_model{num}.h5")
del model
model = load_model(f"models/dense_model{num}.h5")


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
    file.write(f"\tLoss:\t\t{loss_test}\n")
    file.write(f"\tAccuracy:\t{accuracy_test}\n")
    file.write(f"\tRoc_auc:\t{roc_auc_score_test}\n")

experiments = "experiments_dense"
hyperpar = ['', 'epochs', 'units', 'activation', 'loss', 'acc', 'roc-auc']
exp_hyperpar = [epochs, units, activation, loss_test, accuracy_test, roc_auc_score_test]
df = add_experiment(num, exp_hyperpar, experiments, hyperpar)
save_experiments(df, experiments)

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
predictions_train[predictions_train <= 0.5] = 0
predictions_train[predictions_train > 0.5] = 1
sigmoid = np.copy(predictions_test)
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1

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
plt.plot(predictions_test)
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

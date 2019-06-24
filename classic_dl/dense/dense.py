import os

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras import callbacks
import sys
sys.path.append("....")
from utils.load_data import load_data
from utils.utils import add_experiment, save_experiments, generate_indices, model_evaluation,\
                        experiment_results_summary, generate_prediction_plots


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(0)

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
MODEL_PATH = "models/"
model.save(f"{MODEL_PATH}dense_model{num}.h5")
# del model
# model = load_model(f"{MODEL_PATH}dense_model{num}.h5")

# -----------------------------------------------------------------------------
# RESULTS EVALUATION
# -----------------------------------------------------------------------------
""" Predictions on training data """
print("Predicting values on training data...")
predictions_train = model.predict(X_train, batch_size=batch_size).flatten()
loss_train, accuracy_train, roc_auc_train, recall_train = model_evaluation(predictions=predictions_train,
                                                                           y=y_train)
print("Results on training data")
print(f"\tLoss:    \t{loss_train:.4f}")
print(f"\tAccuracy:\t{accuracy_train:.4f}")
print(f"\tROC-AUC: \t{roc_auc_train:.4f}")
print(f"\tRecall:  \t{recall_train:.4f}")

""" Predictions on test data """
print("Predicting values on test data...")
predictions_test = model.predict(X_test, batch_size=batch_size).flatten()
loss_test, accuracy_test, roc_auc_test, recall_test = model_evaluation(predictions=predictions_test,
                                                                       y=y_test)
print("Results on test data")
print(f"\tLoss:    \t{loss_test:.4f}")
print(f"\tAccuracy:\t{accuracy_test:.4f}")
print(f"\tROC-AUC: \t{roc_auc_test:.4f}")
print(f"\tRecall:  \t{recall_test:.4f}")

# -----------------------------------------------------------------------------
# EXPERIMENT RESULTS SUMMARY
# -----------------------------------------------------------------------------
RESULTS_PATH = f"results/{file_name}"
title = "DENSE NEURAL NETWORK"
shapes = {
    "X_train": X_train.shape,
    "y_train": y_train.shape,
    "X_test": X_test.shape,
    "y_test": y_test.shape
}
parameters = {
    "epochs": epochs,
    "batch_size": batch_size,
    "units": units,
    "reg_n": "l2(5e-4)",
    "activation": activation,
    "class_weight": str(class_weight),
}
results_train = {
    "loss_train": loss_train,
    "accuracy_train": accuracy_train,
    "roc_auc_train": roc_auc_train,
    "recall_train": recall_train
}
results_test = {
    "loss_test": loss_test,
    "accuracy_test": accuracy_test,
    "roc_auc_test": roc_auc_test,
    "recall_test": recall_test
}
string_list = []
model.summary(print_fn=lambda x: string_list.append(x))
summary = "\n".join(string_list)

experiment_results_summary(RESULTS_PATH, num, title, summary, shapes, parameters, results_train, results_test)

EXP_FILENAME = "experiments_dense"
hyperpar = ['', 'epochs', 'units', 'activation', 'loss', 'acc', 'roc-auc']
exp_hyperpar = [epochs, units, activation, loss_test, accuracy_test, roc_auc_test]
df = add_experiment(EXP_FILENAME, num, hyperpar, exp_hyperpar)
save_experiments(EXP_FILENAME, df)

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

K.clear_session()
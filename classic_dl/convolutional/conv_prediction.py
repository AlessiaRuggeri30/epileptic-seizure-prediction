import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from keras import callbacks
from conv_model import build_conv_model
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

""" Neural network hyperparameters """
num = 1

epochs = 10
batch_size = 64
depth_conv = 2
depth_dense = 1
filters = 128
kernel_size = 5
reg_n = '5e-1'
reg = l2(float(reg_n))
activation = 'relu'
batch_norm = True
dropout = 0.4
class_weight = {0: (len(y_train) / n_negative), 1: (len(y_train) / n_positive)}

""" Generate sequences """
look_back = 200
stride = 1  # Keep this =1 so that you keep all positive samples
predicted_timestamps = 1
subsampling_factor = 2

target_steps_ahead = [1]  # starting from the position len(sequence)

original_X_train = X_train
original_y_train = y_train
original_X_test = X_test
original_y_test = y_test

for steps_ahead in target_steps_ahead:

    # Generate sequences by computing indices for training data
    inputs_indices_seq, target_indices_seq =  \
        generate_indices([original_y_train],                              # Targets associated to X_train (same shape[0])
                         look_back,                              # Length of input sequences
                         stride=stride,                          # Stride between windows
                         target_steps_ahead=steps_ahead,  # How many steps ahead to predict (x[t], ..., x[t+T] -> y[t+T+k])
                         subsample=True,                         # Whether to subsample
                         subsampling_factor=subsampling_factor   # Keep this many negative samples w.r.t. to positive ones
                         )
    X_train = original_X_train[inputs_indices_seq]
    y_train = original_y_train[target_indices_seq]

    # Generate sequences by computing indices for test data
    inputs_indices_seq, target_indices_seq =  \
        generate_indices([original_y_test],                              # Targets associated to X_train (same shape[0])
                         look_back,                              # Length of input sequences
                         stride=stride,                          # Stride between windows
                         target_steps_ahead=steps_ahead,  # How many steps ahead to predict (x[t], ..., x[t+T] -> y[t+T+k])
                         )
    X_test = original_X_test[inputs_indices_seq]
    y_test = original_y_test[target_indices_seq]

    """ Shuffle training data """
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # -----------------------------------------------------------------------------
    # MODEL BUILDING, TRAINING AND TESTING
    # -----------------------------------------------------------------------------
    """ Build the model """
    exp = "exp" + str(num)
    file_name = exp + "_conv_pred.txt"
    print(f"\n{exp}\n")

    model = build_conv_model(depth_conv, depth_dense, filters, kernel_size, reg, activation, batch_norm, dropout)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    """ Fit the model """
    cb = [
        callbacks.TensorBoard(log_dir=f".logs/pred_logs/{exp}"),
    ]
    model.fit(X_train_shuffled, y_train_shuffled,
              batch_size=batch_size,
              epochs=epochs,
              class_weight=class_weight,
              callbacks=cb)

    """ Save and reload the model """
    model.save(f"models_prediction/conv_pred_model{num}.h5")
    # del model
    # model = load_model(f"models_prediction/conv_pred_model{num}.h5")

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

    with open(f"results_prediction/{file_name}", 'w') as file:
        file.write(f"EXPERIMENT {num}: CONVOLUTIONAL NEURAL NETWORK\n\n")

        file.write("Parameters\n")
        file.write(f"\tepochs:\t\t\t{epochs}\n")
        file.write(f"\tbatch_size:\t\t{batch_size}\n")
        file.write(f"\tdepth_conv:\t\t{depth_conv}\n")
        file.write(f"\tdepth_dense:\t{depth_dense}\n")
        file.write(f"\tfilters:\t\t{filters}\n")
        file.write(f"\tkernel_size:\t{kernel_size}\n")
        file.write(f"\treg:\t\t\tl2({reg_n})\n")
        file.write(f"\tactivation:\t\t{activation}\n")
        file.write(f"\tbatch_norm:\t\t{str(batch_norm)}\n")
        file.write(f"\tdropout:\t\t{dropout}\n")
        file.write(f"\tclass_weight:\t{str(class_weight)}\n")
        file.write(f"\tlook_back:\t\t{look_back}\n")
        file.write(f"\tstride:\t\t\t{stride}\n")
        file.write(f"\tpredicted_timestamps:\t{predicted_timestamps}\n")
        file.write(f"\ttarget_steps_ahead:\t\t{steps_ahead}\n")
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

    experiments = "experiments_conv_pred"
    hyperpar = ['', 'epochs', 'depth_conv', 'depth_dense', 'filters', 'kernel_size', 'activation',
                'l2_reg', 'batch_norm', 'dropout', 'look_back', 'target_steps_ahead',
                'subsampling_factor', 'loss', 'acc', 'roc-auc']
    exp_hyperpar = [epochs, depth_conv, depth_dense, filters, kernel_size, activation,
                    reg_n, batch_norm, dropout, look_back, steps_ahead,
                    subsampling_factor, loss_test, accuracy_test, roc_auc_score_test]
    df = add_experiment(num, exp_hyperpar, experiments, hyperpar)
    save_experiments(df, experiments)

    # -----------------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------------

    plt.subplot(2, 1, 1)
    plt.plot(y_train)
    plt.subplot(2, 1, 2)
    plt.plot(predictions_train)
    plt.savefig(f"./plots_prediction/{exp}_pred-predictions_train.png")
    plt.close()

    plt.subplot(2, 1, 1)
    plt.plot(y_test)
    plt.subplot(2, 1, 2)
    plt.plot(predictions_test)
    plt.savefig(f"./plots_prediction/{exp}_pred-predictions.png")
    plt.close()

    num += 1
    K.clear_session()

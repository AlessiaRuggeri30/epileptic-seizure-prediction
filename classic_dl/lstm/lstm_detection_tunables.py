import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras import callbacks
from lstm_model import build_lstm_model
import sys
sys.path.append("....")
from utils.load_data import load_data
from utils.utils import add_experiment, save_experiments, generate_indices, model_evaluation,\
                        experiment_results_summary, generate_prediction_plots

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(0)

X, y, dataset, seizure = load_data()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


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
num = 170

epochs = 10
batch_size = 64
depth_lstm = [1, 2]
depth_dense = [1, 2]
units_lstm = [128, 256]
reg_n = ['5e-3', '5e-2', '5e-1']
activation = ['tanh', 'relu']
batch_norm = [False, True]
dropout = [0.4, 0.5, 0.6]
class_weight = {0: (len(y_train) / n_negative), 1: (len(y_train) / n_positive)}

# tunables = [depth_lstm, depth_dense, units_lstm, reg_n, activation, batch_norm, dropout]
tunables = [depth_lstm, depth_dense, units_lstm, reg_n, activation, batch_norm, dropout]

for depth_lstm, depth_dense, units_lstm, reg_n, activation, batch_norm, dropout in product(*tunables):
    exp = "exp" + str(num)
    file_name = exp + "_lstm.txt"
    print(f"\n{exp}\n")

    reg = l2(float(reg_n))

    model = build_lstm_model(depth_lstm, depth_dense, units_lstm, reg, activation, batch_norm, dropout)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    """ Fit the model """
    cb = [
        callbacks.TensorBoard(log_dir=f".logs/det_logs/{exp}"),
    ]
    model.fit(X_train_shuffled, y_train_shuffled,
              batch_size=batch_size,
              epochs=epochs,
              class_weight=class_weight,
              callbacks=cb)

    """ Save and reload the model """
    MODEL_PATH = "models/models_detection/"
    model.save(f"{MODEL_PATH}lstm_model{num}.h5")
    # del model
    # model = load_model(f"{MODEL_PATH}lstm_model{num}.h5")

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
    RESULTS_PATH = f"results/results_detection/{file_name}"
    title = "LSTM NEURAL NETWORK"
    shapes = {
        "X_train": X_train.shape,
        "y_train": y_train.shape,
        "X_test": X_test.shape,
        "y_test": y_test.shape
    }
    parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "depth_lstm": depth_lstm,
        "depth_dense": depth_dense,
        "units_lstm": units_lstm,
        "reg_n": f"l2({reg_n})",
        "activation": activation,
        "batch_norm": str(batch_norm),
        "dropout": dropout,
        "class_weight": str(class_weight),
        "look_back": look_back,
        "stride": stride,
        "predicted_timestamps": predicted_timestamps,
        "target_steps_ahead": target_steps_ahead,
        "subsampling_factor": subsampling_factor
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

    EXP_FILENAME = "experiments_lstm_det"
    hyperpar = ['', 'epochs', 'depth_lstm', 'depth_dense', 'units_lstm', 'activation',
                'l2_reg', 'batch_norm', 'dropout', 'look_back', 'target_steps_ahead',
                'subsampling_factor', 'loss', 'acc', 'roc-auc']
    exp_hyperpar = [epochs, depth_lstm, depth_dense, units_lstm, activation,
                    reg_n, batch_norm, dropout, look_back, target_steps_ahead,
                    subsampling_factor, loss_test, accuracy_test, roc_auc_test]
    df = add_experiment(EXP_FILENAME, num, hyperpar, exp_hyperpar)
    save_experiments(EXP_FILENAME, df)

    # -----------------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------------
    PLOTS_PATH = "./plots/plots_detection/"

    PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_pred-predictions_train.png"
    generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_train, y=y_train)

    PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_pred-predictions.png"
    generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_test, y=y_test)

    num += 1
    K.clear_session()

import os

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras import callbacks
from conv_model import build_conv_model
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

""" Neural network hyperparameters """
num = 157

epochs = [10]
batch_size = 64
depth_conv = [3]     # search
depth_dense = 2
filters = [64]
kernel_size = [3]    # search
reg_n = ['5e-1']      #['5e-3', '5e-2', '5e-1']
activation = 'relu'
batch_norm = True
dropout = [0.4]        #[0.5, 0.4, 0.3]
pooling = True
pool_size = 2
padding = 'causal'
dilation_rate = [3]
class_weight = {0: (len(y_train) / n_negative), 1: (len(y_train) / n_positive)}

""" Generate sequences """
look_back = [200]
stride = [10]
predicted_timestamps = 1
subsampling_factor = 2
target_steps_ahead = [2000, 5000]  # starting from the position len(sequence)

original_X_train = X_train
original_y_train = y_train
original_X_test = X_test
original_y_test = y_test

tunables = [epochs, depth_conv, filters, kernel_size, reg_n, dropout, stride, look_back, target_steps_ahead, dilation_rate]

for epochs, depth_conv, filters, kernel_size, reg_n, dropout, stride, look_back, target_steps_ahead, dilation_rate in product(*tunables):
    reg = l2(float(reg_n))

    # Generate sequences by computing indices for training data
    inputs_indices_seq, target_indices_seq =  \
        generate_indices([original_y_train],                              # Targets associated to X_train (same shape[0])
                         look_back,                              # Length of input sequences
                         stride=stride,                          # Stride between windows
                         target_steps_ahead=target_steps_ahead,  # How many steps ahead to predict (x[t], ..., x[t+T] -> y[t+T+k])
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
                         target_steps_ahead=target_steps_ahead,  # How many steps ahead to predict (x[t], ..., x[t+T] -> y[t+T+k])
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

    input_shape = (X_train.shape[-2], X_train.shape[-1])
    model = build_conv_model(depth_conv, depth_dense, filters,
                             kernel_size, reg, activation,
                             batch_norm, dropout, input_shape,
                             pooling, pool_size, padding, dilation_rate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    """ Fit the model """
    # cb = [
    #     callbacks.TensorBoard(log_dir=f".logs/pred_dilated_logs/{exp}"),
    # ]
    # model.fit(X_train_shuffled, y_train_shuffled,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           class_weight=class_weight,
    #           callbacks=cb)

    model.fit(X_train_shuffled, y_train_shuffled,
              batch_size=batch_size,
              epochs=epochs,
              class_weight=class_weight)

    """ Save and reload the model """
    MODEL_PATH = "models/models_prediction_dilated/"
    model.save(f"{MODEL_PATH}conv_pred_model{num}.h5")
    # del model
    # model = load_model(f"{MODEL_PATH}conv_pred_model{num}.h5")

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
    print(f"\tRecall:\t{recall_train:.4f}")

    """ Predictions on test data """
    print("Predicting values on test data...")
    predictions_test = model.predict(X_test, batch_size=batch_size).flatten()
    loss_test, accuracy_test, roc_auc_test, recall_test = model_evaluation(predictions=predictions_test,
                                                                           y=y_test)
    print("Results on test data")
    print(f"\tLoss:    \t{loss_test:.4f}")
    print(f"\tAccuracy:\t{accuracy_test:.4f}")
    print(f"\tROC-AUC: \t{roc_auc_test:.4f}")
    print(f"\tRecall:\t{recall_test:.4f}")

    # -----------------------------------------------------------------------------
    # EXPERIMENT RESULTS SUMMARY
    # -----------------------------------------------------------------------------
    RESULTS_PATH = f"results/results_prediction_dilated/{file_name}"
    title = "CONVOLUTIONAL NEURAL NETWORK"
    shapes = {
        "X_train": X_train.shape,
        "y_train": y_train.shape,
        "X_test": X_test.shape,
        "y_test": y_test.shape
    }
    parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "depth_conv": depth_conv,
        "depth_dense": depth_dense,
        "filters": filters,
        "kernel_size": kernel_size,
        "reg_n": reg_n,
        "activation": activation,
        "batch_norm": str(batch_norm),
        "dropout": dropout,
        "pooling": pooling,
        "pool_size": pool_size,
        "padding": padding,
        "dilation_rate": dilation_rate,
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

    EXP_FILENAME = "experiments_conv_pred_dilated"
    hyperpar = ['', 'epochs', 'depth_conv', 'depth_dense', 'filters', 'kernel_size', 'activation',
                'l2_reg', 'batch_norm', 'dropout', 'pooling', 'pool_size', 'padding',
                'dilation_rate', 'stride', 'look_back', 'target_steps_ahead',
                'subsampling_factor', 'loss', 'acc', 'roc-auc']
    exp_hyperpar = [epochs, depth_conv, depth_dense, filters, kernel_size, activation,
                    reg_n, batch_norm, dropout, pooling, pool_size, padding,
                    dilation_rate, stride, look_back, target_steps_ahead,
                    subsampling_factor, loss_test, accuracy_test, roc_auc_test]
    df = add_experiment(EXP_FILENAME, num, hyperpar, exp_hyperpar)
    save_experiments(EXP_FILENAME, df)

    # -----------------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------------

    PLOTS_PATH = "./plots/plots_prediction_dilated/"

    PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_pred-predictions_train.png"
    generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_train, y=y_train)

    PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_pred-predictions.png"
    generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_test, y=y_test)

    num += 1
    K.clear_session()

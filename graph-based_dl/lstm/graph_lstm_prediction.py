import os
import time

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sys
sys.path.append("....")
from utils.load_data import load_data
from utils.utils import add_experiment, save_experiments, generate_indices, model_evaluation,\
                        experiment_results_summary, generate_prediction_plots, train_test_split,\
                        apply_generate_sequences, data_standardization, compute_class_weight,\
                        generate_graphs
sys.path.append("..")
from graph_model import build_graph_based_lstm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(42)

""" Global parameters """
cross_val = False
saving = True
num = 33

""" Neural network hyperparameters """
epochs = [150]
batch_size = 32
depth_lstm = [1]
depth_dense = [1]
units_lstm = [256]
g_filters = [32]
reg_n = ['5e-4']
activation = ['relu']
batch_norm = False    # Keep it always False, since adding it leads to inconsistent results
dropout = [0.4]
learning_rate = [1e-3]

""" Functional connectivity hyperparameters """
band_freq = (70., 100.)
sampling_freq = [500.]
samples_per_graph = [500]
# fc_measure = 'corr'
link_cutoff = 0.
percentiles = (40, 60)
# band_freq_hi = (20., 45.)
# nfft = 128
# n_overlap = 64
# nf_mode = 'mean'
# self_loops = True

""" Sequences hyperparameters """
subsampling_factor = [1]
stride = [3]
look_back = [5000]
target_steps_ahead = [2000]  # starting from the position len(sequence)
predicted_timestamps = 1

""" Set tunables """
tunables_sequences = [sampling_freq, samples_per_graph, subsampling_factor,
                      stride, look_back, target_steps_ahead]
tunables_network = [epochs, depth_lstm, depth_dense, units_lstm, g_filters, reg_n,
                    activation, dropout, learning_rate]

# -----------------------------------------------------------------------------
# DATA PREPROCESSING
# -----------------------------------------------------------------------------
""" Import dataset """
X, y, dataset, seizure = load_data()

""" Select training set and test set """
X_train_fold, y_train_fold, X_test_fold, y_test_fold = train_test_split(X, y, cross_val=cross_val)
n_folds = len(X_train_fold)

""" Iterate through fold-sets """
for fold in range(n_folds):
    fold_set = fold if cross_val else '/'
    if cross_val: print(f"Fold set: {fold_set}")

    X_train = X_train_fold[fold]
    y_train = y_train_fold[fold]
    X_test = X_test_fold[fold]
    y_test = y_test_fold[fold]

    class_weight = compute_class_weight(y_train)

    original_X_train = X_train
    original_y_train = y_train
    original_X_test = X_test
    original_y_test = y_test

    """ Iterate through sequences/graphs parameters """
    for sampling_freq, samples_per_graph, subsampling_factor,\
        stride, look_back, target_steps_ahead in product(*tunables_sequences):

        """ Generate subsampled sequences """
        X_train, y_train, X_test, y_test = \
            apply_generate_sequences(original_X_train, original_y_train,
                                     original_X_test, original_y_test,
                                     look_back, target_steps_ahead,
                                     stride, subsampling_factor)

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        """ Generate graphs from sequences """
        # slice_n = 100       # TODO: remove slice and apply on whole dataset
        # X_train = X_train[0:slice_n]
        # y_train = y_train[0:slice_n]
        # X_test = X_test[0:slice_n]
        # y_test = y_test[0:slice_n]

        start = time.time()
        X_train, A_train, E_train = generate_graphs(X_train, band_freq, sampling_freq, samples_per_graph, percentiles)
        X_test, A_test, E_test = generate_graphs(X_test, band_freq, sampling_freq, samples_per_graph, percentiles)
        end = time.time()
        interval = end - start
        print(f"All sequences converted. Spent time:   {int(interval)} sec (~{round(interval/60)} min)\n")

        """ Standardize data """
        X_train, X_test = data_standardization(X_train, X_test)

        print(f"X_train: {X_train.shape}\t\tX_test: {X_test.shape}")
        print(f"A_train: {A_train.shape}\t\tA_test: {A_test.shape}")
        print(f"E_train: {E_train.shape}\t\tE_test: {E_test.shape}")

        """ Iterate through network parameters """
        for epochs, depth_lstm, depth_dense, units_lstm, g_filters, reg_n, activation,\
            dropout, learning_rate in product(*tunables_network):
            # -----------------------------------------------------------------------------
            # MODEL BUILDING, TRAINING AND TESTING
            # -----------------------------------------------------------------------------
            exp = "exp" + str(num)
            file_name = exp + "_conv_pred.txt"
            print(f"\n{exp}\n")

            reg = l2(float(reg_n))

            F = X_train.shape[-1]
            N = A_train.shape[-1]
            S = E_train.shape[-1]
            seq_length = int(look_back/samples_per_graph)

            """ Build the model """
            model = build_graph_based_lstm(F, N, S, seq_length,
                                           depth_lstm, depth_dense, units_lstm, g_filters,
                                           reg, activation, batch_norm, dropout)
            optimizer = Adam(learning_rate)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            model.fit([X_train, A_train, E_train], y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      class_weight=class_weight)

            if saving:
                """ Save and reload the model """
                MODEL_PATH = "models/models_prediction/"
                model.save(f"{MODEL_PATH}graph_lstm_pred_model{num}.h5")
                # del model
                # model = load_model(f"{MODEL_PATH}conv_pred_model{num}.h5")

            # -----------------------------------------------------------------------------
            # RESULTS EVALUATION
            # -----------------------------------------------------------------------------
            """ Predictions on training data """
            print("Predicting values on training data...")
            loss_train_keras, accuracy_train_keras = model.evaluate([X_train, A_train, E_train], y_train, batch_size=batch_size)
            predictions_train = model.predict([X_train, A_train, E_train], batch_size=batch_size).flatten()
            loss_train, accuracy_train, roc_auc_train, recall_train = model_evaluation(predictions=predictions_train,
                                                                                       y=y_train)
            print("Results on training data")
            print(f"\tLoss:    \t{loss_train:.4f}")
            print(f"\tAccuracy:\t{accuracy_train:.4f}")
            print(f"\tROC-AUC: \t{roc_auc_train:.4f}")
            print(f"\tRecall:  \t{recall_train:.4f}")

            """ Predictions on test data """
            print("Predicting values on test data...")
            loss_test_keras, accuracy_test_keras = model.evaluate([X_test, A_test, E_test], y_test, batch_size=batch_size)
            predictions_test = model.predict([X_test, A_test, E_test], batch_size=batch_size).flatten()
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
            if saving:
                RESULTS_PATH = f"results/results_prediction/{file_name}"
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
                    "g_filters": g_filters,
                    "reg_n": f"l2({reg_n})",
                    "activation": activation,
                    "batch_norm": str(batch_norm),
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "class_weight": str(class_weight),
                    "look_back": look_back,
                    "stride": stride,
                    "predicted_timestamps": predicted_timestamps,
                    "target_steps_ahead": target_steps_ahead,
                    "subsampling_factor": subsampling_factor,
                    "band_freq": band_freq,
                    "sampling_freq": sampling_freq,
                    "samples_per_graph": samples_per_graph,
                    "link_cutoff": link_cutoff,
                    "percentiles": percentiles
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

                EXP_FILENAME = "experiments_lstm_pred"
                hyperpar = ['', 'epochs', 'depth_lstm', 'depth_dense', 'units_lstm', 'g_filters',
                            'activation', 'l2_reg', 'batch_norm', 'dropout', 'stride', 'subsampling_factor',
                            'samples_per_graph', 'look_back', 'target_steps_ahead', 'fold_set',
                            'loss', 'acc', 'roc-auc', 'recall']
                exp_hyperpar = [epochs, depth_lstm, depth_dense, units_lstm, g_filters,
                                activation, reg_n, batch_norm, dropout, stride, subsampling_factor,
                                samples_per_graph, look_back, target_steps_ahead, fold_set,
                                f"{loss_test:.5f}", f"{accuracy_test:.5f}", f"{roc_auc_test:.5f}", f"{recall_test:.5f}"]
                df = add_experiment(EXP_FILENAME, num, hyperpar, exp_hyperpar)
                save_experiments(EXP_FILENAME, df)

                # -----------------------------------------------------------------------------
                # PLOTS
                # -----------------------------------------------------------------------------
                PLOTS_PATH = "./plots/plots_prediction/"

                PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_pred-predictions_train.png"
                generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_train, y=y_train, moving_a=100)

                PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_pred-predictions.png"
                generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_test, y=y_test, moving_a=100)

            num += 1
            K.clear_session()


#     TODO: mean of results of model applied to 3 folds

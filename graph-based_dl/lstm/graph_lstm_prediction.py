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

""" Import dataset """
X, y, dataset, seizure = load_data()
cross_val = False

""" Select training set and test set """
X_train_fold, y_train_fold, X_test_fold, y_test_fold = train_test_split(X, y, cross_val=cross_val)
n_folds = len(X_train_fold)

""" Neural network hyperparameters """
num = 1

epochs = [10]
batch_size = 64
depth_lstm = [1]     # search
depth_dense = [2]
units_lstm = [256]
g_filters = [32]
reg_n = ['5e-1']      #['5e-3', '5e-2', '5e-1']
activation = ['relu']
batch_norm = [True]
dropout = [0.4]        #[0.5, 0.4, 0.3]
learning_rate = [1e-3]

""" Functional connectivity hyperparameters """
band_freq = (70., 100.)
sampling_freq = [500.]
samples_per_graph = [500]
# fc_measure = 'corr'
# link_cutoff = 0.
percentiles = (40, 60)
# band_freq_hi = (20., 45.)
# nfft = 128
# n_overlap = 64
# nf_mode = 'mean'
# self_loops = True

""" Sequences hyperparameters """
subsampling_factor = [2]
stride = [10]
look_back = [5000]
target_steps_ahead = [2000]  # starting from the position len(sequence)
predicted_timestamps = 1

""" Set tunables """
tunables_sequences = [sampling_freq, samples_per_graph, subsampling_factor,
                      stride, look_back, target_steps_ahead]
tunables_network = [epochs, depth_lstm, depth_dense, units_lstm, g_filters, reg_n,
                    activation, batch_norm, dropout, learning_rate]

""" Iterate through fold-sets """
for fold in range(n_folds):
    fold_set = fold if cross_val else '/'
    # -----------------------------------------------------------------------------
    # DATA PREPROCESSING
    # -----------------------------------------------------------------------------
    X_train = X_train_fold[fold]
    y_train = y_train_fold[fold]
    X_test = X_test_fold[fold]
    y_test = y_test_fold[fold]

    class_weight = compute_class_weight(y_train)

    """ Standardize data """
    X_train, X_test = data_standardization(X_train, X_test)

    original_X_train = X_train
    original_y_train = y_train
    original_X_test = X_test
    original_y_test = y_test

    """ Iterate through sequences/graphs parameters """
    for sampling_freq, samples_per_graph, subsampling_factor,\
        stride, look_back, target_steps_ahead in product(*tunables_sequences):

        """ Generate subsampled sequences """
        X_train, y_train, X_test, Y_test = \
            apply_generate_sequences(X_train, y_train, X_test, y_test, look_back, target_steps_ahead,
                                     stride, subsampling_factor)

        """ Shuffle training data """
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        """ Generate graphs from sequences """
        seq = X_train_shuffled[0:10]
        y_train_shuffled = y_train_shuffled[0:10]
        start = time.time()
        X, A, E = generate_graphs(seq, band_freq, sampling_freq, samples_per_graph, percentiles)
        end = time.time()
        print(f"X: {X.shape}")
        print(f"A: {A.shape}")
        print(f"E: {E.shape}")
        print(end - start)

        """ Iterate through network parameters """
        for epochs, depth_lstm, depth_dense, units_lstm, g_filters, reg_n, activation,\
            batch_norm, dropout, learning_rate in product(*tunables_network):
            # -----------------------------------------------------------------------------
            # MODEL BUILDING, TRAINING AND TESTING
            # -----------------------------------------------------------------------------
            exp = "exp" + str(num)
            file_name = exp + "_conv_pred.txt"
            print(f"\n{exp}\n")

            reg = l2(float(reg_n))

            F = X.shape[-1]
            N = A.shape[-1]
            S = E.shape[-1]
            seq_length = int(look_back/samples_per_graph)

            """ Build the model """
            model = build_graph_based_lstm(F, N, S, seq_length,
                               depth_lstm, depth_dense, units_lstm, g_filters,
                               reg, activation, batch_norm, dropout)
            optimizer = Adam(learning_rate)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            model.fit([X, A, E], y_train_shuffled,
                      batch_size=batch_size,
                      epochs=epochs,
                      class_weight=class_weight)

            # """ Save and reload the model """
            # MODEL_PATH = "models/models_prediction/"
            # model.save(f"{MODEL_PATH}graph_lstm_pred_model{num}.h5")
            # # del model
            # # model = load_model(f"{MODEL_PATH}conv_pred_model{num}.h5")

            # -----------------------------------------------------------------------------
            # RESULTS EVALUATION
            # -----------------------------------------------------------------------------
            """ Predictions on training data """
            print("Predicting values on training data...")
            predictions_train = model.predict([X, A, E], batch_size=batch_size).flatten()
            loss_train, accuracy_train, roc_auc_train, recall_train = model_evaluation(predictions=predictions_train,
                                                                                       y=y_train_shuffled)
            print("Results on training data")
            print(f"\tLoss:    \t{loss_train:.4f}")
            print(f"\tAccuracy:\t{accuracy_train:.4f}")
            print(f"\tROC-AUC: \t{roc_auc_train:.4f}")
            print(f"\tRecall:  \t{recall_train:.4f}")

            # """ Predictions on test data """
            # print("Predicting values on test data...")
            # predictions_test = model.predict(X_test, batch_size=batch_size).flatten()
            # loss_test, accuracy_test, roc_auc_test, recall_test = model_evaluation(predictions=predictions_test,
            #                                                                        y=y_test)
            # print("Results on test data")
            # print(f"\tLoss:    \t{loss_test:.4f}")
            # print(f"\tAccuracy:\t{accuracy_test:.4f}")
            # print(f"\tROC-AUC: \t{roc_auc_test:.4f}")
            # print(f"\tRecall:  \t{recall_test:.4f}")

#     TODO: insert fold_set to experiment values in the table
#     TODO: mean of results of model applied to 3 folds

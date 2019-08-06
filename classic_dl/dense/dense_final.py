import os
import time

import tensorflow.keras.backend as K
import numpy as np
from itertools import product
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
import sys
sys.path.append("....")
from utils.load_data import load_data
from utils.utils import add_experiment, save_experiments, model_evaluation,\
                        experiment_results_summary, generate_prediction_plots, train_test_split,\
                        data_standardization, compute_class_weight

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(42)

""" Global parameters """
cross_val = True
saving = True
num = 37

""" Neural network hyperparameters """
epochs = [20]
batch_size = 32
units = [512]
reg_n = ['5e-1']
activation = ['relu', 'tanh']
batch_norm = True
dropout = [0.4, 0.5]

""" Set tunables """
tunables_network = [epochs, units, reg_n, activation, dropout]

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

    """ Standardize data """
    X_train, X_test = data_standardization(X_train, X_test)

    original_X_train = X_train
    original_y_train = y_train
    original_X_test = X_test
    original_y_test = y_test

    """ Shuffle training data """
    X_train_shuffled, y_train_shuffled = shuffle(original_X_train, original_y_train)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    """ Iterate through network parameters """
    for epochs, units, reg_n, activation, dropout in product(*tunables_network):
        # -----------------------------------------------------------------------------
        # MODEL BUILDING, TRAINING AND TESTING
        # -----------------------------------------------------------------------------
        exp = "exp" + str(num)
        file_name = exp + "_dense_det.txt"
        print(f"\n{exp}\n")

        reg = l2(float(reg_n))

        """ Build the model """
        model = Sequential()
        model.add(Dense(units, activation=activation, kernel_regularizer=reg, batch_input_shape=(batch_size, 90)))
        model.add(Dropout(dropout))
        model.add(Dense(units, activation=activation, kernel_regularizer=reg))
        model.add(Dropout(dropout))
        model.add(Dense(int(units/2), activation=activation, kernel_regularizer=reg))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X_train_shuffled, y_train_shuffled,
                  batch_size=batch_size,
                  epochs=epochs,
                  class_weight=class_weight)

        if saving:
            """ Save and reload the model """
            MODEL_PATH = "models/models_detection_final/"
            model.save(f"{MODEL_PATH}dense_det_model{num}.h5")
            # del model
            # model = load_model(f"{MODEL_PATH}dense_pred_model{num}.h5")

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
        if saving:
            RESULTS_PATH = f"results/results_detection_final/{file_name}"
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
                "reg_n": f"l2({reg_n})",
                "activation": activation,
                "batch_norm": str(batch_norm),
                "dropout": dropout,
                "class_weight": str(class_weight)
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

            EXP_FILENAME = "experiments_dense_det_final"
            hyperpar = ['', 'epochs', 'units', 'activation', 'l2_reg', 'batch_norm', 'dropout',
                        'fold_set', 'loss', 'acc', 'roc-auc', 'recall']
            exp_hyperpar = [epochs, units, activation, reg_n, batch_norm, dropout,
                            fold_set,
                            f"{loss_test:.5f}", f"{accuracy_test:.5f}", f"{roc_auc_test:.5f}", f"{recall_test:.5f}"]
            df = add_experiment(EXP_FILENAME, num, hyperpar, exp_hyperpar)
            save_experiments(EXP_FILENAME, df)

            # -----------------------------------------------------------------------------
            # PLOTS
            # -----------------------------------------------------------------------------
            PLOTS_PATH = "./plots/plots_detection_final/"

            PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_det-predictions_train.png"
            generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_train, y=y_train, moving_a=100)

            PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_det-predictions.png"
            generate_prediction_plots(PLOTS_FILENAME, predictions=predictions_test, y=y_test, moving_a=100)

        num += 1
        K.clear_session()

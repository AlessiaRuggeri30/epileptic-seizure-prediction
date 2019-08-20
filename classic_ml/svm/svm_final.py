import os
from joblib import dump, load

import numpy as np
from itertools import product
from sklearn import svm
from sklearn.metrics import brier_score_loss
from sklearn.utils import shuffle
import sys
sys.path.append("....")
from utils.load_data import load_data
from utils.utils import add_experiment, save_experiments, model_evaluation, data_standardization,\
                        experiment_results_summary, generate_prediction_plots, train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(42)

""" Global parameters """
cross_val = True
saving = True
num = 7

""" Model hyperparameters """
gamma = 'scale'
weighted = [True, False]
random_state = 42

""" Set tunables """
tunables_model = [weighted]

# -----------------------------------------------------------------------------
# DATA PREPROCESSING
# -----------------------------------------------------------------------------
""" Import dataset """
X, y, dataset, seizure = load_data(reduced=True)

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

    if weighted:
        class_weight = 'balanced'
    else:
        class_weight = None

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
    for weighted in product(*tunables_model):
        # -----------------------------------------------------------------------------
        # MODEL BUILDING, TRAINING AND TESTING
        # -----------------------------------------------------------------------------
        exp = "exp" + str(num)
        file_name = exp + "_svm.txt"
        print(f"\n{exp}\n")

        """ Build the model """
        clf = svm.SVC(gamma=gamma, probability=True, class_weight=class_weight, verbose=True)

        clf.fit(X_train_shuffled, y_train_shuffled)

        if saving:
            """ Save and reload the model """
            MODEL_PATH = "models_final/"
            dump(clf, f"{MODEL_PATH}svm_model{num}.joblib")
            # clf = load(f"svm_model{num}.joblib")

        # -----------------------------------------------------------------------------
        # RESULTS EVALUATION
        # -----------------------------------------------------------------------------
        """ Predictions on training data """
        print("Predicting values on training data...")
        probabilities_train = clf.predict_proba(X_train)[:, 1]
        brier_loss_train = brier_score_loss(y_train, probabilities_train)
        loss_train, accuracy_train, roc_auc_train, recall_train = model_evaluation(predictions=probabilities_train,
                                                                                   y=y_train)
        print("Results on training data")
        print(f"\tBrier loss:    \t{brier_loss_train:.4f}")
        print(f"\tLoss:    \t{loss_train:.4f}")
        print(f"\tAccuracy:\t{accuracy_train:.4f}")
        print(f"\tROC-AUC: \t{roc_auc_train:.4f}")
        print(f"\tRecall:  \t{recall_train:.4f}")

        """ Predictions on test data """
        print("Predicting values on test data...")
        probabilities_test = clf.predict_proba(X_test)[:, 1]
        brier_loss_test = brier_score_loss(y_test, probabilities_test)
        loss_test, accuracy_test, roc_auc_test, recall_test = model_evaluation(predictions=probabilities_test,
                                                                               y=y_test)
        print("Results on test data")
        print(f"\tBrier loss:    \t{brier_loss_test:.4f}")
        print(f"\tLoss:    \t{loss_test:.4f}")
        print(f"\tAccuracy:\t{accuracy_test:.4f}")
        print(f"\tROC-AUC: \t{roc_auc_test:.4f}")
        print(f"\tRecall:  \t{recall_test:.4f}")

        # -----------------------------------------------------------------------------
        # EXPERIMENT RESULTS SUMMARY
        # -----------------------------------------------------------------------------
        if saving:
            RESULTS_PATH = f"results_final/{file_name}"
            title = "SVM"
            shapes = {
                "X_train": X_train.shape,
                "y_train": y_train.shape,
                "X_test": X_test.shape,
                "y_test": y_test.shape
            }
            parameters = {
                "gamma": gamma,
                "random_state": random_state,
                "weighted": weighted,
                "class_weight": str(class_weight)
            }
            results_train = {
                "brier_loss_train": brier_loss_train,
                "loss_train": loss_train,
                "accuracy_train": accuracy_train,
                "roc_auc_train": roc_auc_train,
                "recall_train": recall_train
            }
            results_test = {
                "brier_loss_test": brier_loss_test,
                "loss_test": loss_test,
                "accuracy_test": accuracy_test,
                "roc_auc_test": roc_auc_test,
                "recall_test": recall_test
            }

            summary = f"Num of svm for each class: {str(clf.n_support_)}"
            experiment_results_summary(RESULTS_PATH, num, title, summary, shapes, parameters, results_train, results_test)

            EXP_FILENAME = "experiments_svm"
            hyperpar = ['', 'gamma', 'weighted', 'fold_set',
                        'brier_loss', 'loss', 'acc', 'roc-auc', 'recall']
            exp_hyperpar = [gamma, weighted, fold_set, f"{brier_loss_test:.5f}",
                            f"{loss_test:.5f}", f"{accuracy_test:.5f}", f"{roc_auc_test:.5f}", f"{recall_test:.5f}"]
            df = add_experiment(EXP_FILENAME, num, hyperpar, exp_hyperpar)
            save_experiments(EXP_FILENAME, df)

            # -----------------------------------------------------------------------------
            # PLOTS
            # -----------------------------------------------------------------------------
            PLOTS_PATH = "./plots_final/"

            PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_det-predictions_train.png"
            generate_prediction_plots(PLOTS_FILENAME, predictions=probabilities_train, y=y_train, moving_a=100)

            PLOTS_FILENAME = f"{PLOTS_PATH}{exp}_det-predictions.png"
            generate_prediction_plots(PLOTS_FILENAME, predictions=probabilities_test, y=y_test, moving_a=100)

        num += 1

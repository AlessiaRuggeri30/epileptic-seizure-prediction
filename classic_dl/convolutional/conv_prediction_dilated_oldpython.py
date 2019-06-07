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
num = 66

epochs = 10
batch_size = 64
depth_conv = [3, 4, 5]     # search
depth_dense = 2
filters = [64, 128]
kernel_size = [3]    # search
reg_n = ['5e-2']      #['5e-3', '5e-2', '5e-1']
activation = 'relu'
batch_norm = True
dropout = [0.5]        #[0.5, 0.4, 0.3]
pooling = True
pool_size = 2
padding = 'causal'
dilation_rate = [3]
class_weight = {0: (len(y_train) / n_negative), 1: (len(y_train) / n_positive)}

""" Generate sequences """
look_back = [5000]
stride = [10]
predicted_timestamps = 1
subsampling_factor = 2
target_steps_ahead = [2000]  # starting from the position len(sequence)

original_X_train = X_train
original_y_train = y_train
original_X_test = X_test
original_y_test = y_test

tunables = [depth_conv, filters, kernel_size, reg_n, dropout, stride, look_back, target_steps_ahead, dilation_rate]

for depth_conv, filters, kernel_size, reg_n, dropout, stride, look_back, target_steps_ahead, dilation_rate in product(*tunables):
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
    print("\n{}\n".format(exp))

    input_shape = (X_train.shape[-2], X_train.shape[-1])
    model = build_conv_model(depth_conv, depth_dense, filters,
                             kernel_size, reg, activation,
                             batch_norm, dropout, input_shape,
                             pooling, pool_size, padding, dilation_rate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    """ Fit the model """
    # cb = [
    #     callbacks.TensorBoard(log_dir=".logs/pred_dilated_logs/{}".format(exp)),
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
    model.save("models/models_prediction_dilated/conv_pred_model{}.h5".format(num))
    # del model
    # model = load_model("models/models_prediction_dilated/conv_pred_model{}.h5".format(num))

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
    print("\tLoss:    \t{:.4f}".format(loss_train))
    print("\tAccuracy:\t{:.4f}".format(accuracy_train))
    print("\tROC-AUC: \t{:.4f}".format(roc_auc_score_train))

    """ Predictions on test data """
    loss_keras, metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("\tLoss:    \t{:.4f}".format(loss_keras))
    print("\tAccuracy:\t{:.4f}".format(metrics))

    print("Predicting values on test data...")
    predictions_test = model.predict(X_test, batch_size=batch_size).flatten()

    print("Results on test data")
    loss_test = log_loss(y_test, predictions_test, eps=1e-7)  # for the clip part, eps=1e-15 is too small for float32
    accuracy_test = accuracy_score(y_test, np.round(predictions_test))
    roc_auc_score_test = roc_auc_score(y_test, predictions_test)
    print("\tLoss:    \t{:.4f}".format(loss_test))
    print("\tAccuracy:\t{:.4f}".format(accuracy_test))
    print("\tROC-AUC: \t{:.4f}".format(roc_auc_score_test))

    # -----------------------------------------------------------------------------
    # EXPERIMENT RESULTS SUMMARY
    # -----------------------------------------------------------------------------
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    summary = "\n".join(string_list)

    with open("results/results_prediction_dilated/{}".format(file_name), 'w') as file:
        file.write("EXPERIMENT {}: CONVOLUTIONAL NEURAL NETWORK\n\n".format(num))

        file.write("NO DROPOUT OR KERNEL REGULARIZATION BETWEEN CONVOLUTIONAL LAYERS\n")
        file.write("Increased number of samples: 150000 instead of 100000\n\n")

        file.write("Parameters\n")
        file.write("\tepochs:\t\t\t{}\n".format(epochs))
        file.write("\tbatch_size:\t\t{}\n".format(batch_size))
        file.write("\tdepth_conv:\t\t{}\n".format(depth_conv))
        file.write("\tdepth_dense:\t{}\n".format(depth_dense))
        file.write("\tfilters:\t\t{}\n".format(filters))
        file.write("\tkernel_size:\t{}\n".format(kernel_size))
        file.write("\treg:\t\t\tl2({})\n".format(reg_n))
        file.write("\tactivation:\t\t{}\n".format(activation))
        file.write("\tbatch_norm:\t\t{}\n".format(str(batch_norm)))
        file.write("\tdropout:\t\t{}\n".format(dropout))
        file.write("\tclass_weight:\t{}\n".format(str(class_weight)))
        file.write("\tlook_back:\t\t{}\n".format(look_back))
        file.write("\tstride:\t\t\t{}\n".format(stride))
        file.write("\tpredicted_timestamps:\t{}\n".format(predicted_timestamps))
        file.write("\ttarget_steps_ahead:\t\t{}\n".format(target_steps_ahead))
        file.write("\tsubsampling_factor:\t\t{}\n\n".format(subsampling_factor))

        file.write("Model\n")
        file.write("{}\n\n".format(summary))

        file.write("Data shape\n")
        file.write("\tX_train shape:\t{}\n".format(X_train.shape))
        file.write("\ty_train shape:\t{}\n".format(y_train.shape))
        file.write("\tX_test shape: \t{}\n".format(X_test.shape))
        file.write("\ty_test shape: \t{}\n\n".format(y_test.shape))

        file.write("Results on train set\n")
        file.write("\tLoss:\t\t{}\n".format(loss_train))
        file.write("\tAccuracy:\t{}\n".format(accuracy_train))
        file.write("\tRoc_auc:\t{}\n\n".format(roc_auc_score_train))

        file.write("Results on test set\n")
        file.write("\tLoss_keras:\t{}\n".format(loss_keras))
        file.write("\tLoss:\t\t{}\n".format(loss_test))
        file.write("\tAccuracy:\t{}\n".format(accuracy_test))
        file.write("\tRoc_auc:\t{}\n".format(roc_auc_score_test))

    experiments = "experiments_conv_pred_dilated"
    hyperpar = ['', 'epochs', 'depth_conv', 'depth_dense', 'filters', 'kernel_size', 'activation',
                'l2_reg', 'batch_norm', 'dropout', 'pooling', 'pool_size', 'padding',
                'dilation_rate', 'stride', 'look_back', 'target_steps_ahead',
                'subsampling_factor', 'loss', 'acc', 'roc-auc']
    exp_hyperpar = [epochs, depth_conv, depth_dense, filters, kernel_size, activation,
                    reg_n, batch_norm, dropout, pooling, pool_size, padding,
                    dilation_rate, stride, look_back, target_steps_ahead,
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
    plt.savefig("./plots/plots_prediction_dilated/{}_pred-predictions_train.png".format(exp))
    plt.close()

    plt.subplot(2, 1, 1)
    plt.plot(y_test)
    plt.subplot(2, 1, 2)
    plt.plot(predictions_test)
    plt.savefig("./plots/plots_prediction_dilated/{}_pred-predictions.png".format(exp))
    plt.close()

    num += 1
    K.clear_session()

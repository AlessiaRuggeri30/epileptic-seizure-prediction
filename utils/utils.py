import numpy as np
import pandas as pd
import os.path
from spektral.brain import get_fc
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, recall_score
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# DATA PROCESSING
# -----------------------------------------------------------------------------

def generate_sequences(inputs, targets, length, target_steps_ahead=0,
                       sampling_rate=1, stride=1, start_index=0, shuffle=False,
                       epochs=1, batch_size=32, subsample=False,
                       subsampling_cutoff_threshold=0.5, subsampling_factor=1.):
    """
    Takes a time series and its associated targets and yields batches of
    sub-sequences and their target.
    :param subsampling_factor: if `balanced=True`, keep
    `n_positive * subsampling_factor` negative samples.
    :param subsampling_cutoff_threshold: consider targets below this value to
    be negative.
    :param inputs: list of numpy arrays (more than one input is possible)
    :param targets: list of numpy arrays (more than one target is possible)
    :param length: length of the input windows
    :param target_steps_ahead: delay of the target w.r.t. the associated
    sequence. If the sequence is `input[i:i+length]`, the target will be
    `target[i+length+target_steps_ahead]`.
    :param sampling_rate: rate at which to sample input sequences, e.g.
    `input[i:i+length:sampling_rate]`.
    :param stride: consecutive sequences will be distant this number of
    timesteps.
    :param start_index: ignore the input before this timestep.
    :param shuffle: shuffle the sequences at every epoch (if `False`, the
    sequences are yielded in temporal order).
    :param epochs: number of epochs to run for.
    :param batch_size: size of a minibatch to be returned by the generator.
    :param subsample: subsample class 0 (based on the first target).
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(targets, list):
        targets = [targets]
    if stride < 1:
        raise ValueError('stride must be greater than 0')

    if batch_size == -1:
        batch_size = np.inf

    inputs_indices_seq, target_indices_seq = \
        generate_indices(targets, length,
                         target_steps_ahead=target_steps_ahead,
                         sampling_rate=sampling_rate, stride=stride,
                         start_index=start_index, subsample=subsample,
                         subsampling_cutoff_threshold=subsampling_cutoff_threshold,
                         subsampling_factor=subsampling_factor)

    n_batches_full = int(target_indices_seq.shape[0] // batch_size)
    n_residual_samples = int(target_indices_seq.shape[0] % batch_size)

    n_batches_full = n_batches_full + 1 if n_residual_samples > 0 else n_batches_full
    yield n_batches_full  # Yield this for keras

    for e in range(epochs):
        if shuffle:
            perm = np.random.permutation(np.arange(target_indices_seq.shape[0]))
            inputs_indices_seq = inputs_indices_seq[perm]
            target_indices_seq = target_indices_seq[perm]
        for b in range(n_batches_full):
            strt = b * batch_size
            stop = strt + batch_size
            iis = inputs_indices_seq[strt:stop]
            tis = target_indices_seq[strt:stop]
            output_sequences = [i_[iis] for i_ in inputs]
            output_targets = [t_[tis] for t_ in targets]
            yield (output_sequences, output_targets)

        # Last samples
        if n_residual_samples > 0:
            strt = target_indices_seq.shape[0] - n_residual_samples
            iis = inputs_indices_seq[strt:]
            tis = target_indices_seq[strt:]
            output_sequences = [i_[iis] for i_ in inputs]
            output_targets = [t_[tis] for t_ in targets]
            yield (output_sequences, output_targets)


def generate_indices(targets, length, target_steps_ahead=0,
                     sampling_rate=1, stride=1, start_index=0,
                     subsample=False, subsampling_cutoff_threshold=0.5,
                     subsampling_factor=1.):
    len_data = targets[0].shape[0]
    start_index = start_index + length
    number_of_sequences = (len_data - start_index - target_steps_ahead) // stride
    end_index = start_index + number_of_sequences * stride + target_steps_ahead

    if start_index > end_index:
        raise ValueError('`start_index+length=%i > end_index=%i` '
                         'is disallowed, as no part of the sequence '
                         'would be left to be used as current step.'
                         % (start_index, end_index))

    inputs_indices_seq = [np.arange(idx - length, idx, sampling_rate)
                          for idx in np.arange(start_index, end_index - target_steps_ahead + 1, stride)]
    inputs_indices_seq = np.array(inputs_indices_seq)
    target_indices_seq = np.arange(start_index + target_steps_ahead - 1, end_index, stride)

    if subsample:
        # Which indices correspond to positive/negative samples
        # (i.e., meta-indices to select the actual indices)
        positive_meta_idxs = np.where(targets[0][target_indices_seq] >= subsampling_cutoff_threshold)[0]
        negative_meta_idxs = np.where(targets[0][target_indices_seq] < subsampling_cutoff_threshold)[0]

        # Number of positive/negative samples to keep (keep all positive)
        n_positive_class = positive_meta_idxs.shape[0]
        n_negative_class_keep = min(
            int(n_positive_class * subsampling_factor),
            negative_meta_idxs.shape[0]
        )

        # Select the actual meta-indices for positve (all) and negative (random
        # choice among all the possible)
        negative_meta_idxs_keep = np.random.choice(negative_meta_idxs,
                                                   n_negative_class_keep,
                                                   replace=False)

        # Stack the meta-indices and preserve temporal order of the sequences.
        # There will be gaps in the data but this is ok
        total_meta_idxs_keep = np.hstack((positive_meta_idxs, negative_meta_idxs_keep))
        total_meta_idxs_keep = np.sort(total_meta_idxs_keep)

        # Filter the actual indices with the meta indices
        inputs_indices_seq = inputs_indices_seq[total_meta_idxs_keep]
        target_indices_seq = target_indices_seq[total_meta_idxs_keep]

    return inputs_indices_seq, target_indices_seq


def generate_graphs(seq, band_freq, sampling_freq, samples_per_graph, percentiles):
    seq = np.transpose(seq, (0, 2, 1))

    X = []
    A = []
    E = []
    for i in range(seq.shape[0]):
        # print(f"Single sequence: {seq[i].shape}")
        adj, nf, ef = get_fc(seq[i], band_freq, sampling_freq, samples_per_graph,
                             percentiles=percentiles)
        # print(f"adj: {adj.shape}")
        # print(f"nf: {nf.shape}")
        # print(f"ef: {ef.shape}")
        # print(adj[adj > 0])
        X.append(np.expand_dims(nf, axis=-1))
        A.append(adj)
        E.append(ef)
    print()
    X = np.asarray(X)
    A = np.asarray(A)
    E = np.asarray(E)

    return X, A, E


# -----------------------------------------------------------------------------
# STORING EXPERIMENTS RESULTS
# -----------------------------------------------------------------------------

def create_experiments(hyperpar):
    data = np.array([hyperpar])
    df = pd.DataFrame(data=data[1:, 1:],
                      index=data[1:, 0],
                      columns=data[0, 1:])
    return df


def add_experiment(filename, num, hyperpar, exp_hyperpar):
    if not os.path.isfile(filename):
        df = create_experiments(hyperpar=hyperpar)
    else:
        df = pd.read_pickle(filename)
    df.loc["exp{}".format(num)] = exp_hyperpar
    return df


def save_experiments(filename, dataframe):
    dataframe.to_pickle(filename)
    dataframe.to_csv("{}.csv".format(filename))


def model_evaluation(predictions, y):
    loss = log_loss(y, predictions, eps=1e-7)  # for the clip part, eps=1e-15 is too small for float32
    accuracy = accuracy_score(y, np.round(predictions))
    roc_auc = roc_auc_score(y, predictions)
    recall = recall_score(y, np.round(predictions))

    return loss, accuracy, roc_auc, recall


def experiment_results_summary(path, num, title, summary, shapes, parameters, results_train, results_test):
    with open(path, 'w') as file:
        file.write(f"EXPERIMENT {num}: {title}\n\n")

        file.write("Parameters\n")
        for key, value in parameters.items():
            file.write(f"\t{key}:   {value}\n")
        file.write("\n")
        file.write("Model\n")
        file.write(f"{summary}\n")
        file.write("\n")
        file.write("Data shape\n")
        for key, value in shapes.items():
            file.write(f"\t{key}:   {value}\n")
        file.write("\n")
        file.write("Results on train set\n")
        for key, value in results_train.items():
            file.write(f"\t{key}:   {value}\n")
        file.write("\n")
        file.write("Results on test set\n")
        for key, value in results_test.items():
            file.write(f"\t{key}:   {value}\n")
        file.write("\n")


def generate_prediction_plots(filename, predictions, y):
    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.subplot(2, 1, 2)
    plt.plot(predictions)
    plt.savefig(filename)
    plt.close()


# -----------------------------------------------------------------------------
# TEST CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # pass

    # x_1 = np.arange(1, 14)
    # x_2 = np.arange(101, 114)
    # y_1 = np.arange(1, 14)
    # y_2 = np.arange(101, 114)
    # data = generate_sequences([x_1, x_2], [y_1, y_2], 4, target_steps_ahead=2,
    #                           stride=1, batch_size=1, shuffle=False, subsample=True)
    # print(data.__next__())
    # for (x_1_, x_2_), (y_1_, y_2_) in data:
    #     print('batch')
    #     print(x_1_)
    #     # print(x_2_)
    #     print(y_1_)
    #     # print(y_2_)

    filename = "experiments"
    hyperpar = ['', 'par1', 'par2', 'par3', 'par4']
    num = 5
    exp_hyperpar = [10, 11, 12, 13]
    df = add_experiment(num, exp_hyperpar, filename, hyperpar)

    print(df)

    save_experiments(df, filename)


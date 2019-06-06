import numpy as np


def load_data():
    """ Variables """
    n_clip = 3
    X = {}
    y = {}
    # Trimming around the centered seizure
    interval = {1: {'start': 1590000, 'end': 1740000},
                2: {'start': 100000, 'end': 250000},
                3: {'start': 10000, 'end': 160000}}
    # Trimming so that the interval finishes exactly with the end of the seizure
    # interval = {1: {'start': 1599381, 'end': 1699381},
    #             2: {'start': 101013, 'end': 201013},
    #             3: {'start': 10699, 'end': 110699}}
    seizure = {1: {'start': 1684381 - interval[1]['start'], 'end': 1699381 - interval[1]['start']},
               2: {'start': 188013 - interval[2]['start'], 'end': 201013 - interval[2]['start']},
               3: {'start': 96699 - interval[3]['start'], 'end': 110699 - interval[3]['start']}}

    """ Load datasets """
    for c in range(1, n_clip + 1):
        path = "/home/phait/datasets/ieeg/TWH056/clips/TWH056_Day-504_Clip-0-{}.npz".format(c)  # server
        # path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

        with np.load(path) as data:
            data = dict(data)

        data['szr_bool'] = data['szr_bool'].astype(int)  # one-hot encoding

        X[c] = data['ieeg'].T[interval[c]['start']:interval[c]['end']]
        y[c] = data['szr_bool'][interval[c]['start']:interval[c]['end']]

        if c == 1:
            dataset = X[c]
        else:
            dataset = np.concatenate((dataset, X[c]), axis=0)

    return X, y, dataset, seizure


# X[c]      samples from clip c
# y[c]      targets from clip c
# dataset   concatenation of samples from all clips (concatenation of all X[c], used for scaling on entire dataset)

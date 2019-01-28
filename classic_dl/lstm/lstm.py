import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn import preprocessing
# import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Input, Embedding, Dense, Dropout, LSTM, InputLayer, Reshape

""" Variables """
n_clip = 3
X = {}
y = {}
interval = {1: {'start': 1640000, 'end': 1740000},
            2: {'start': 150000, 'end': 250000},
            3: {'start': 60000, 'end': 160000}}
seizure = {1: {'start': 1684381 - interval[1]['start'], 'end': 1699381 - interval[1]['start']},
           2: {'start': 188013 - interval[2]['start'], 'end': 201013 - interval[2]['start']},
           3: {'start': 96699 - interval[3]['start'], 'end': 110699 - interval[3]['start']}}


""" Load datasets """
for c in range(1, n_clip + 1):
    path = f"/home/phait/datasets/ieeg/TWH056_Day-504_Clip-0-{c}.npz"  # server
    # path = "../../dataset/TWH056_Day-504_Clip-0-1.npz"                     # local

    with np.load(path) as data:
        data = dict(data)

    data['szr_bool'] = data['szr_bool'].astype(int)  # one-hot encoding

    X[c] = data['ieeg'].T[interval[c]['start']:interval[c]['end']]
    y[c] = data['szr_bool'][interval[c]['start']:interval[c]['end']]


""" Select training set and test set """
X_train = np.concatenate((X[2], X[3]), axis=0)
y_train = np.concatenate((y[2], y[3]), axis=0)
X_test = X[1]
y_test = y[1]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# print(X_train[0:5])


""" Build the model """
epochs = 20
batch_size = 64

model = Sequential()
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.add(InputLayer(batch_input_shape=(batch_size, None, 90)))
# model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5, stateful=True))
# model.summary()

""" Fit the model """
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

""" Save and reload the model """
model.save('lstm_model.h5')
del model
model = load_model('lstm_model.h5')

""" Predictions on test data """
loss, metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Loss: {loss}")
print(f"Accuracy: {metrics}")

print("Predicting values on test data...")
predictions = model.predict(X_test, batch_size=batch_size)
predictions = predictions.reshape(-1)
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1
errors = abs(predictions - y_test)

print("Results")
loss = round(log_loss(y_test, predictions, eps=1e-7), 4)    # for the clip part, eps=1e-15 is too small for float32
accuracy = round(accuracy_score(y_test, predictions), 4)
roc_auc_score = round(roc_auc_score(y_test, predictions), 4)
print(f"\tLoss:\t\t{loss}")
print(f"\tAccuracy:\t{accuracy}")
print(f"\tRoc:\t\t{roc_auc_score}")


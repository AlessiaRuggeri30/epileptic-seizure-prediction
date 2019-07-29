from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential


def build_lstm_model(depth_lstm, depth_dense, units_lstm, reg, activation, batch_norm, dropout):
    model = Sequential()
    # lstm layers
    for i in range(depth_lstm):
        if i != 0:
            if batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
        if i == (depth_lstm-1):
            model.add(LSTM(units_lstm, activation='tanh', kernel_regularizer=reg))
        else:
            model.add(LSTM(units_lstm, activation='tanh', kernel_regularizer=reg,
                           return_sequences=True))
    # dense layers
    for k in range(depth_dense):
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout))
        if k == (depth_dense-1):
            model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg))
        else:
            model.add(Dense(256, activation=activation, kernel_regularizer=reg))
    return model

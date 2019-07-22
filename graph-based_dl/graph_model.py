import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, LSTM, BatchNormalization, Dropout, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from spektral.layers import EdgeConditionedConv, GlobalAvgPool


def build_graph_based_lstm(F, N, S, seq_length,
                           depth_lstm, depth_dense, units_lstm, g_filters,
                           reg, activation, batch_norm, dropout):

    X_in = Input(shape=(seq_length, N, F))
    A_in = Input(shape=(seq_length, N, N))
    E_in = Input(shape=(seq_length, N, N, S))

    X_td = Lambda(lambda x: K.reshape(x, (-1, N, F)))(X_in)
    A_td = Lambda(lambda x: K.reshape(x, (-1, N, N)))(A_in)
    E_td = Lambda(lambda x: K.reshape(x, (-1, N, N, S)))(E_in)

    """ Graph Convolution-Pooling block """
    ecc = EdgeConditionedConv(g_filters,
                              kernel_network=[32, 32],
                              activation='relu',
                              kernel_regularizer=reg,
                              use_bias=True)

    td = ecc([X_td, A_td, E_td])
    pool = GlobalAvgPool()(td)

    """ LSTM block """
    lstm = Lambda(lambda x: K.reshape(x, (-1, seq_length, K.int_shape(x)[-1])))(pool)  # Reshape to sequences
    for i in range(depth_lstm):
        if i != 0:
            if batch_norm:
                lstm = BatchNormalization()(lstm)
            lstm = Dropout(dropout)(lstm)
        if i == (depth_lstm-1):
            lstm = LSTM(units_lstm, activation='tanh', kernel_regularizer=reg)(lstm)
        else:
            lstm = LSTM(units_lstm, activation='tanh', kernel_regularizer=reg,
                        return_sequences=True)(lstm)

    """ Dense block """
    dense = lstm
    for k in range(depth_dense):
        if batch_norm:
            dense = BatchNormalization()(dense)
        dense = Dropout(dropout)(dense)
        if k == (depth_dense-1):
            dense = Dense(1, activation='sigmoid', kernel_regularizer=reg)(dense)
        else:
            dense = Dense(128, activation=activation, kernel_regularizer=reg)(dense)

    """ Build model """
    model = Model(inputs=[X_in, A_in, E_in], outputs=dense)

    return model


def build_graph_based_conv(F, N, S, seq_length,
                           depth_conv, depth_dense, filters, kernel_size,
                           g_filters, reg, activation,
                           batch_norm, dropout,
                           pooling=True, pool_size=2,
                           padding='valid', dilation_rate=1):

    half_filters = int(filters / 2)

    X_in = Input(shape=(seq_length, N, F))
    A_in = Input(shape=(seq_length, N, N))
    E_in = Input(shape=(seq_length, N, N, S))

    X_td = Lambda(lambda x: K.reshape(x, (-1, N, F)))(X_in)
    A_td = Lambda(lambda x: K.reshape(x, (-1, N, N)))(A_in)
    E_td = Lambda(lambda x: K.reshape(x, (-1, N, N, S)))(E_in)

    """ Graph Convolution-Pooling block """
    ecc = EdgeConditionedConv(g_filters,
                              kernel_network=[32, 32],
                              activation='relu',
                              kernel_regularizer=reg,
                              use_bias=True)

    td = ecc([X_td, A_td, E_td])
    pool = GlobalAvgPool()(td)

    """ Conv block """
    conv = Lambda(lambda x: K.reshape(x, (-1, seq_length, K.int_shape(x)[-1])))(pool)  # Reshape to sequences
    for i in range(depth_conv):
        if i != 0:
            if batch_norm:
                conv = BatchNormalization()(conv)
            conv = Dropout(dropout)(conv)
        if i == (depth_conv-1):
            conv = Conv1D(filters=half_filters, kernel_size=kernel_size, activation=activation,
                          kernel_regularizer=reg, padding=padding, dilation_rate=dilation_rate)(conv)
        else:
            conv = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation,
                          kernel_regularizer=reg, padding=padding, dilation_rate=dilation_rate)(conv)
        if pooling:
            conv = MaxPooling1D(pool_size=pool_size)(conv)

    conv = Flatten()(conv)

    """ Dense block """
    dense = conv
    for k in range(depth_dense):
        if batch_norm:
            dense = BatchNormalization()(dense)
        dense = Dropout(dropout)(dense)
        if k == (depth_dense-1):
            dense = Dense(1, activation='sigmoid', kernel_regularizer=reg)(dense)
        else:
            dense = Dense(128, activation=activation, kernel_regularizer=reg)(dense)

    """ Build model """
    model = Model(inputs=[X_in, A_in, E_in], outputs=dense)

    return model
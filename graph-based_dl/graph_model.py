# TODO: implement the graph-based model both for the conv and the lstm

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, TimeDistributed, LSTM, BatchNormalization, Dropout, Dense
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

    td = TimeDistributed(ecc)([X_td, A_td, E_td])
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


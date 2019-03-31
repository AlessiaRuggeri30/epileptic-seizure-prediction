from keras.layers import Dense, Dropout, Conv1D, BatchNormalization, Flatten, MaxPooling2D
from keras.models import Sequential


def build_conv_model(depth_conv, depth_dense, filters, kernel_size, reg, activation, batch_norm, dropout):
    model = Sequential()
    # conv + pooling layers
    for i in range(depth_conv):
        if i != 0:
            if batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
        if i == (depth_conv-1):
            if i == 0:
                model.add(Conv1D(filters, kernel_size=64, activation=activation, kernel_regularizer=reg,
                                 input_shape=(200, 90)))
            else:
                model.add(Conv1D(filters, kernel_size=64, activation=activation, kernel_regularizer=reg))
        else:
            if i == 0:
                model.add(Conv1D(filters, kernel_size=kernel_size, activation=activation, kernel_regularizer=reg,
                                 input_shape=(200, 90)))
            else:
                model.add(Conv1D(filters, kernel_size=kernel_size, activation=activation, kernel_regularizer=reg))
        model.add(MaxPooling2D(pool_size=(2, 2)))
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

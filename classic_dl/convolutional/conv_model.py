from tensorflow.keras.layers import Dense, Dropout, Conv1D, BatchNormalization, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential


def build_conv_model(depth_conv, depth_dense, filters,
                     kernel_size, reg, activation,
                     batch_norm, dropout, input_shape,
                     pooling=True, pool_size=2,
                     padding='valid', dilation_rate=1):

    half_filters = int(filters/2)
    model = Sequential()

    # conv + pooling layers
    for i in range(depth_conv):
        if i == (depth_conv-1):
            if i == 0:
                model.add(Conv1D(filters=half_filters, kernel_size=kernel_size, activation=activation,
                                 kernel_regularizer=reg, padding=padding, dilation_rate=dilation_rate,
                                 input_shape=input_shape))
            else:
                model.add(Conv1D(filters=half_filters, kernel_size=kernel_size, activation=activation,
                                 kernel_regularizer=reg, padding=padding, dilation_rate=dilation_rate))
        else:
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation,
                                 kernel_regularizer=reg, padding=padding, dilation_rate=dilation_rate,
                                 input_shape=input_shape))
            else:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation,
                                 kernel_regularizer=reg, padding=padding, dilation_rate=dilation_rate))
        if pooling:
            model.add(MaxPooling1D(pool_size=pool_size))

    model.add(Flatten())

    # dense layers
    for k in range(depth_dense):
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout))
        if k == (depth_dense-1):
            model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg))
        else:
            model.add(Dense(128, activation=activation, kernel_regularizer=reg))
    return model

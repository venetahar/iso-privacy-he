from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf


def conv_model(input_shape, kernel_size, stride, out_channels, avg_pool_size, dense_units, num_classes):
    """
    Creates a convolutional model.
    """
    input_x = Input(input_shape)
    x = Conv2D(out_channels, kernel_size=kernel_size, activation='relu', strides=stride, padding='valid')(input_x)
    x = AveragePooling2D(pool_size=avg_pool_size)(x)
    x_shape = x.get_shape()[1:]
    x = tf.reshape(x, [-1, np.prod(x_shape)])

    for dense_unit in dense_units:
        x = Dense(dense_unit, activation='relu')(x)
    output = Dense(num_classes, name="output")(x)

    return Model(inputs=input_x, outputs=output)

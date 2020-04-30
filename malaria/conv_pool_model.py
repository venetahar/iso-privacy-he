import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Input


def conv_pool_model(input_shape, kernel_sizes, stride, channels, avg_pool_sizes, dense_units, num_classes):
    """
    Returns a ConvModel.
    :param input_shape: The input shape.
    :param kernel_sizes: The kernel sizes.
    :param stride: The stride.
    :param channels: The channels.
    :param avg_pool_sizes: The pool sizes.
    :param dense_units: The dense units.
    :param num_classes: The number of classes.
    """
    input_x = Input(input_shape, name="input")
    x = Conv2D(channels[0], kernel_size=kernel_sizes[0], activation='relu', strides=stride, padding='valid')(input_x)

    x = AveragePooling2D(pool_size=avg_pool_sizes[0])(x)
    x = Conv2D(channels[1], kernel_size=kernel_sizes[1], activation='relu', strides=stride, padding='valid')(x)
    x = AveragePooling2D(pool_size=avg_pool_sizes[1])(x)

    x_shape = x.get_shape()[1:]
    x = tf.reshape(x, [-1, np.prod(x_shape)])

    for dense_unit in dense_units:
        x = Dense(dense_unit, activation='relu')(x)

    output = Dense(num_classes, name="output")(x)

    return Model(inputs=input_x, outputs=output)

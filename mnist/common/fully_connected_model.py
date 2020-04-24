import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input


def fully_connected_model(input_shape, dense_units, num_classes):
    """
    Returns a FullyConnectedModel.
    :param input_shape: The input shape.
    :param dense_units: The dense units.
    :param num_classes: The number of classes.
    """
    input_x = Input(input_shape, name="input")
    flatten_shape = np.prod(input_shape)
    x = tf.reshape(input_x, [-1, flatten_shape])

    for dense_unit in dense_units:
        x = Dense(dense_unit, activation='relu')(x)

    output = Dense(num_classes, name="output")(x)

    return Model(inputs=input_x, outputs=output)

from tensorflow.keras.layers import Dense, Reshape, Input
from tensorflow.keras.models import Sequential
import numpy as np


class FullyConnectedModel:
    """
    Creates a Fully Connected Model.
    """

    def __init__(self, input_shape, dense_units, num_classes):
        """
        Returns a FullyConnectedModel.
        :param input_shape: The input shape.
        :param dense_units: The dense units.
        :param num_classes: The number of classes.
        """
        self.model = Sequential()
        self.model.add(Input(input_shape, name="input"))
        flatten_shape = np.prod(input_shape)
        self.model.add(Reshape((flatten_shape,), input_shape=input_shape))

        for dense_unit in dense_units:
            self.model.add(
                Dense(dense_unit, activation='relu')
            )

        self.model.add(Dense(num_classes, name="output"))

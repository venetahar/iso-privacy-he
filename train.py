from numpy.random import seed

from malaria.constants import MALARIA_TARGET_DATA_PATH_PREFIX, MALARIA_DATA_PATH, MALARIA_MODEL_PATH, MALARIA_MODEL_NAME
from malaria.malaria_training import train_malaria_model
from mnist.constants import MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME, MNIST_CONV_MODEL_NAME, \
    MNIST_TARGET_DATA_PREFIX

seed(1)
from tensorflow.compat.v1 import set_random_seed

set_random_seed(2)

from common.model_factory import CONV_MODEL_TYPE, FULLY_CONNECTED_MODEL_TYPE
from mnist.mnist_training import train_mnist_model


def train_model(model_type):
    """
    Trains a model.
    :param model_type: The model type.
    """
    if model_type == 'mnist_fc':
        train_mnist_model(FULLY_CONNECTED_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME,
                          MNIST_TARGET_DATA_PREFIX)
    elif model_type == 'mnist_conv':
        train_mnist_model(CONV_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_CONV_MODEL_NAME,
                          MNIST_TARGET_DATA_PREFIX)
    elif model_type == 'malaria_conv':
        train_malaria_model(model_path=MALARIA_MODEL_PATH, model_name=MALARIA_MODEL_NAME,
                            source_data_path=MALARIA_DATA_PATH,
                            target_data_path_prefix=MALARIA_TARGET_DATA_PATH_PREFIX)


train_model('mnist_conv')

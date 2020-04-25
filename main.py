from numpy.random import seed

from malaria.common.malaria_training import train_malaria_model

seed(1)
from tensorflow.compat.v1 import set_random_seed

set_random_seed(2)

from common.model_factory import CONV_MODEL_TYPE, FULLY_CONNECTED_MODEL_TYPE
from mnist.common.mnist_training import train_mnist_model

MNIST_MODEL_PATH = 'he_transformer_code/mnist/models/'
MNIST_FULLY_CONNECTED_MODEL_NAME = 'alice_fc3_model'
MNIST_CONV_MODEL_NAME = 'alice_conv_model'

MALARIA_MODEL_PATH = 'he_transformer_code/malaria/models/'
MALARIA_DATA_PATH = 'malaria/data/cell_images/'
MALARIA_MODEL_NAME = 'alice_conv_pool_model'
MALARIA_TARGET_DATA_PATH_PREFIX = 'he_transformer_code/malaria/data/bob_test_'
MALARIA_BATCHED_TEST_DATA_DIR = 'he_transformer_code/malaria/batched_test_data/'
MALARIA_BATCHED_TEST_DATA_LABELS_DIR = 'he_transformer_code/malaria/batched_test_labels/'
MALARIA_BATCHED_TEST_DATA_FILE_PREFIX = 'bob_test_'


def run_mnist_fully_connected_experiment():
    train_mnist_model(FULLY_CONNECTED_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME,
                      'he_transformer_code/mnist/data/bob_test_')


# run_mnist_fully_connected_experiment()


def run_mnist_conv_experiment():
    train_mnist_model(CONV_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_CONV_MODEL_NAME,
                      'he_transformer_code/mnist/data/bob_test_')


def run_malaria_experiment():
    train_malaria_model(model_path=MALARIA_MODEL_PATH, model_name=MALARIA_MODEL_NAME,
                        source_data_path=MALARIA_DATA_PATH,
                        target_data_path_prefix=MALARIA_TARGET_DATA_PATH_PREFIX)


run_malaria_experiment()

MNIST_WIDTH = 28
MNIST_HEIGHT = 28
NUM_CHANNELS = 1
INPUT_SHAPE = (MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS)

DENSE_UNITS = [128, 128]
NUM_CLASSES = 10

TRAINING_PARAMS = {
        'learning_rate': 0.001,
        'momentum': 0.9,
        'num_epochs': 15,
        'optimizer': 'Adam',
        'batch_size': 128
}

MNIST_NORM_MEAN = [0.1307]
MNIST_NORM_STD = [0.3081]

MNIST_MODEL_PATH = 'mnist/models/'
MNIST_FULLY_CONNECTED_MODEL_NAME = 'alice_fc3_model'
MNIST_CONV_MODEL_NAME = 'alice_conv_model'
MNIST_TARGET_DATA_PREFIX = 'mnist/data/bob_test_'

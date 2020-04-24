from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from common.model_factory import ModelFactory
from common.model_training import ModelTraining
from common.utils.data_utils import DataUtils
from mnist.common.constants import NUM_CLASSES, TRAINING_PARAMS, MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS, \
    INPUT_SHAPE, MNIST_NORM_MEAN, MNIST_NORM_STD


def train_mnist_model(model_type, model_path, model_name, target_data_path_prefix):
    """
    Trains a MNIST model and saves the model graph.
    :param model_name:
    :param model_path: The model path.
    :param model_type: The model type to use.
    """
    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
    training_data, training_labels = preprocess(training_data, training_labels)
    test_data, test_labels = preprocess(test_data, test_labels)

    model = ModelFactory.create_model(model_type, INPUT_SHAPE, NUM_CLASSES)
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train(training_data, training_labels)
    model_training.evaluate_plain_text(test_data, test_labels)
    DataUtils.save_data(test_data, test_labels, target_data_path_prefix)
    DataUtils.save_graph(model, model_path, model_name + '.pb')


def preprocess(data, labels):
    """
    Pre-processes the data.
    :param data: The data to pre-process.
    :param labels: The labels.
    :return: Pre-processed data and labels.
    """
    data = data.reshape(data.shape[0], MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS)
    data = data.astype('float32')
    data = normalize(data)

    labels = to_categorical(labels, NUM_CLASSES)
    return data, labels


def normalize(data):
    return (data/255.0 - MNIST_NORM_MEAN) / MNIST_NORM_STD


def test_saved_model(model_path):
    (_, _), (test_data, test_labels) = mnist.load_data()
    test_data, test_labels = preprocess(test_data, test_labels)

    new_model = DataUtils.load_model(model_path + '.pb')
    new_model.evaluate(test_data, test_labels)

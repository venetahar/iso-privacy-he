import numpy as np
import time
import pyhe_client


def perform_inference(test_data, test_data_labels, parameters):
    num_classes = test_data_labels.shape[1]
    test_data_flat = test_data.flatten("C")

    client = pyhe_client.HESealClient(
        parameters.hostname,
        parameters.port,
        parameters.batch_size,
        {
            parameters.tensor_name: (parameters.encrypt_data_str, test_data_flat)
        })

    print("Waiting for results.")
    start_time = time.time()
    prediction_scores = np.array(client.get_results()).reshape(parameters.batch_size, num_classes)
    end_time = time.time()
    print("Got predictions with shape {} in time: {}".format(prediction_scores.shape, end_time - start_time))
    correct_predictions = calculate_num_correct_predictions(prediction_scores, test_data_labels)
    num_samples = test_data_labels.shape[0]
    print('HE-Transformer: {}/{} Test set: Accuracy: ({:.4f})'.format(correct_predictions, num_samples,
                                                                      correct_predictions / num_samples))
    return correct_predictions


def calculate_num_correct_predictions(prediction_scores, one_hot_labels):
    predictions = prediction_scores.argmax(axis=1)
    labels = np.where(one_hot_labels == 1)[1]
    return np.sum(predictions == labels)

import numpy as np

from argument_parsers import client_argument_parser
from generic_client import perform_inference

if __name__ == "__main__":
    parameters, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Supplied parameters cannot be parsed", unparsed)
        exit(1)

    test_data = np.load('malaria/bob_test_data.npy')
    test_data_labels = np.load('malaria/bob_test_data_labels.npy')

    start_batch = parameters.start_batch
    end_batch = start_batch + parameters.batch_size

    if parameters.batch_mode is True:
        print("Batch mode on")
        num_samples = test_data.shape[0]
        index = 0
        correct_predictions = 0
        while index < num_samples:
            new_index = index + parameters.batch_size if index + parameters.batch_size < num_samples else num_samples
            correct_predictions += perform_inference(test_data[index: new_index], test_data_labels[index: new_index], parameters)
            index = new_index
        print('HE-Transformer: {}/{} Test set: Accuracy: ({:.4f})'.format(correct_predictions, num_samples,
                                                                          correct_predictions / num_samples))
    else:
        print("Batch mode off")
        perform_inference(test_data[start_batch: end_batch], test_data_labels[start_batch: end_batch], parameters)


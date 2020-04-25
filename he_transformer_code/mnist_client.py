import numpy as np

from argument_parsers import client_argument_parser
from generic_client import perform_inference

if __name__ == "__main__":
    parameters, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Supplied parameters cannot be parsed", unparsed)
        exit(1)

    test_data = np.load('mnist/bob_test_data.npy')
    test_data_labels = np.load('mnist/bob_test_data_labels.npy')

    index = 0
    num_samples = test_data.shape[0]
    batch_size = parameters.batch_size
    correct_predictions = 0
    while index < num_samples:
        new_index = index + batch_size if index + batch_size < num_samples else num_samples
        # Needed as the last batch might be of a different size.
        parameters.batch_size = new_index - index
        print(parameters.batch_size)
        correct_predictions += perform_inference(test_data[index: new_index],
                                                 test_data_labels[index: new_index], parameters)
        index = new_index
    print('HE-Transformer: {}/{} Test set: Accuracy: ({:.4f})'.format(correct_predictions, num_samples,
                                                                      correct_predictions / num_samples))

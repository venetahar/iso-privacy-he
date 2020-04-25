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
    perform_inference(test_data[start_batch: end_batch], test_data_labels[start_batch: end_batch], parameters)


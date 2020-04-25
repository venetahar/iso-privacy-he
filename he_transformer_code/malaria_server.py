import numpy as np

from argument_parsers import server_argument_parser
from generic_server import perform_inference

if __name__ == "__main__":
    parameters, unparsed = server_argument_parser().parse_known_args()

    if unparsed:
        print("Unparsed parameters:", unparsed)
        exit(1)
    if parameters.encrypt_server_data and parameters.enable_client:
        raise Exception(
            "encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )
    if parameters.model_file == "":
        raise Exception("parameters.model_file must be set")

    test_data = np.load('malaria/bob_test_data.npy')
    test_data_labels = np.load('malaria/bob_test_data_labels.npy')

    perform_inference(test_data[:parameters.batch_size], test_data_labels[:parameters.batch_size], parameters)

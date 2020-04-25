import numpy as np
import time

import generic_client
import generic_server

from client_parameters import ClientParameters
from server_parameters import ServerParameters

if __name__ == "__main__":
    client_parameters = ClientParameters(batch_size=500)
    server_parameters = ServerParameters(enable_client=True, model_file="models/alice_conv_model.pb",
                                         encryption_parameters="../../configs/he_seal_ckks_config_N13_L8.json")


    test_data = np.load('mnist/bob_test_data.npy')
    test_data_labels = np.load('mnist/bob_test_data_labels.npy')

    index = 0
    num_samples = test_data.shape[0]
    batch_size = client_parameters.batch_size
    correct_predictions = 0
    while index < num_samples:
        new_index = index + batch_size if index + batch_size < num_samples else num_samples
        # Needed as the last batch might be of a different size.
        client_parameters.batch_size = new_index - index
        print(client_parameters.batch_size)
        generic_server.perform_inference(test_data[:server_parameters.batch_size],
                                         test_data_labels[:server_parameters.batch_size], server_parameters)
        print("Sleeping to let the server start")
        time.sleep(60)

        correct_predictions += generic_client.perform_inference(test_data[index: new_index],
                                                                test_data_labels[index: new_index], client_parameters)
        index = new_index
    print('HE-Transformer: {}/{} Test set: Accuracy: ({:.4f})'.format(correct_predictions, num_samples,
                                                                      correct_predictions / num_samples))

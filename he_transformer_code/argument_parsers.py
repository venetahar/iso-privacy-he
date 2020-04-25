import argparse


def client_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size. Default 1.")
    parser.add_argument("--hostname", type=str, default="localhost", help="Hostname of server. Default localhost.")
    parser.add_argument("--port", type=int, default=34000, help="Port of server. Default: 34000")
    parser.add_argument("--encrypt_data_str", type=str, default="encrypt",
                        help='"encrypt" to encrypt client data, "plain" to not encrypt. Default encrypt.', )
    parser.add_argument("--tensor_name", type=str, default="import/input",
                        help="Input tensor name. Default import/input")
    parser.add_argument("--start_batch", type=int, default=0, help="Test data start index. Default 0.")

    return parser


def server_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--enable_client", type=bool, default=False, help="Enable the client")
    parser.add_argument("--enable_gc", type=bool, default=False, help="Enable garbled circuits")
    parser.add_argument("--mask_gc_inputs", type=bool, default=False, help="Mask garbled circuits inputs", )
    parser.add_argument("--mask_gc_outputs", type=bool, default=False, help="Mask garbled circuits outputs", )
    parser.add_argument("--num_gc_threads", type=int, default=1,
                        help="Number of threads to run garbled circuits with", )
    parser.add_argument("--backend", type=str, default="HE_SEAL", help="Name of backend to use")
    parser.add_argument("--encryption_parameters", type=str, default="",
                        help="Filename containing json description of encryption parameters, or json description itself", )
    parser.add_argument("--encrypt_server_data", type=bool, default=False,
                        help="Encrypt server data (should not be used when enable_client is used)", )
    parser.add_argument("--pack_data", type=bool, default=True, help="Use plaintext packing on data")
    parser.add_argument("--start_batch", type=int, default=0, help="Test data start index")
    parser.add_argument("--model_file", type=str, default="", help="Filename of saved protobuf model")
    parser.add_argument("--input_node", type=str, default="import/input:0", help="Tensor name of data input", )
    parser.add_argument("--output_node", type=str, default="import/output/BiasAdd:0",
                        help="Tensor name of model output", )
    parser.add_argument("--num_requests", type=int, default=1, help="Number of request before shutting down the client.")

    return parser

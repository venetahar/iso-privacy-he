import numpy as np
import tensorflow as tf
import ngraph_bridge

from tensorflow.core.protobuf import rewriter_config_pb2

from he_transformer_code.utils import load_pb_model


def perform_inference(test_data, test_labels, parameters):
    tf.import_graph_def(load_pb_model(parameters.model_file))
    print("Loaded model")

    model_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
        parameters.input_node)
    model_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
        parameters.output_node)

    config = build_server_config(parameters, model_input.name)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        prediction_scores = model_output.eval(feed_dict={model_input: test_data})

    if not parameters.enable_client:
        correct_predictions = calculate_num_correct_predictions(prediction_scores, test_labels)
        print('HE-Transformer: Test set: Accuracy: ({:.4f})'.format(correct_predictions / parameters.batch_size))


def calculate_num_correct_predictions(prediction_scores, one_hot_labels):
    predictions = prediction_scores.argmax(axis=1)
    labels = np.where(one_hot_labels == 1)[1]
    return np.sum(predictions == labels)


def build_server_config(parameters, model_input_node_name):
    """
    Builds a server config from parameters. Based on the code examples provided here:
    https://github.com/IntelAI/he-transformer/blob/master/examples/MNIST/mnist_util.py
    :param parameters: The supplied parameters.
    :param model_input_node_name: The input name of the model.
    :return: A config.
    """
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_config.min_graph_nodes = -1
    server_config = rewriter_config.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = parameters.backend.encode()
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map["encryption_parameters"].s = parameters.encryption_parameters.encode()
    server_config.parameter_map["enable_client"].s = str(parameters.enable_client).encode()
    server_config.parameter_map["enable_gc"].s = (str(parameters.enable_gc)).encode()
    server_config.parameter_map["mask_gc_inputs"].s = (str(parameters.mask_gc_inputs)).encode()
    server_config.parameter_map["mask_gc_outputs"].s = (str(parameters.mask_gc_outputs)).encode()
    server_config.parameter_map["num_gc_threads"].s = (str(parameters.num_gc_threads)).encode()

    if parameters.enable_client:
        server_config.parameter_map[model_input_node_name].s = b"client_input"
    elif parameters.encrypt_server_data:
        server_config.parameter_map[model_input_node_name].s = b"encrypt"

    if parameters.pack_data:
        server_config.parameter_map[model_input_node_name].s += b",packed"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_config)))

    return config

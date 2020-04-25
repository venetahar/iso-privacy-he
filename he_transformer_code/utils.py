import tensorflow as tf


def load_pb_model(model_file):
    with tf.io.gfile.GFile(model_file, "rb") as file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(file.read())

    return graph_def

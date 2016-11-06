import tensorflow as tf
from deeprl.helpers.layer_helpers import weight_variable, bias_variable, conv2d


class Convnet:

    def __init__(self, output_size, sess=tf.Session()):
        self.sess = sess
        self.input = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.conv1_weight = weight_variable(shape=[8, 8, 4, 16], name='conv1_weight')
        self.conv1_bias = bias_variable(shape=[16], name='conv1_bias')
        self.conv1_layer = tf.nn.relu(conv2d(self.input, self.conv1_weight, 4) + self.conv1_bias)
        self.conv2_weight = weight_variable(shape=[4, 4, 16, 32], name='conv2_weight')
        self.conv2_bias = bias_variable(shape=[32], name='conv2_bias')
        self.conv2_layer = tf.nn.relu(conv2d(self.conv1_layer, self.conv2_weight, 2) + self.conv2_bias)
        self.conv2_layer_flattened = tf.reshape(self.conv2_layer, [-1, 9*9*32])
        self.fc1_weight = weight_variable(shape=[9*9*32, 256], name='fc1_weight')
        self.fc1_bias = bias_variable(shape=[256], name='fc1_bias')
        self.fc1_layer = tf.nn.relu(tf.matmul(self.conv2_layer_flattened, self.fc1_weight) + self.fc1_bias)
        self.fc2_weight = weight_variable(shape=[256, output_size], name='fc2_weight')
        self.fc2_bias = bias_variable(shape=[output_size], name='fc2_bias')
        self.output = tf.matmul(self.fc1_layer, self.fc2_weight) + self.fc2_bias
        # self.reshaped_output = tf.reshape(self.output, shape=[-1, self.num_predictions+1, self.ACTION_SIZE])
        self.weights = [self.conv1_weight, self.conv1_bias, self.conv2_weight, self.conv2_bias, self.fc1_weight,
                        self.fc1_bias, self.fc2_weight, self.fc2_bias]

        self.sess.run(tf.initialize_all_variables())
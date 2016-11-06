import tensorflow as tf


def weight_variable(shape, name, initial_weight=None):
    if initial_weight:
        return tf.Variable(initial_weight, name=name)
    else:
        initial = tf.random_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)


def bias_variable(shape, name, initial_weight=None):
    if initial_weight is None:
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)
    else:
        return tf.Variable(initial_weight, name=name)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


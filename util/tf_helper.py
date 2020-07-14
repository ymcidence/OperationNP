from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


@tf.function
def row_distance(tensor_a, tensor_b):
    """

    :param tensor_a: [B d2]
    :param tensor_b: [k d2]
    :return: [B k]
    """
    na = tf.reduce_sum(tf.square(tensor_a), 1)
    nb = tf.reduce_sum(tf.square(tensor_b), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    rslt = tf.sqrt(tf.maximum(na - 2 * tf.matmul(tensor_a, tensor_b, False, True) + nb, 0.0))

    return rslt * -1.


@tf.function
def row_distance_cosine(tensor_a, tensor_b):
    norm_a = tf.sqrt(tf.reduce_sum(tf.pow(tensor_a, 2), 1, keepdims=True))  # [N, 1]
    norm_b = tf.sqrt(tf.reduce_sum(tf.pow(tensor_b, 2), 1, keepdims=True))
    denominator = tf.matmul(norm_a, norm_b, transpose_b=True)
    numerator = tf.matmul(tensor_a, tensor_b, transpose_b=True)

    return numerator / denominator


@tf.function
def row_distance_hamming(tensor_in):
    """
    Hamming-distance-based graph. It is self-connected.
    :param tensor_in: [N D]
    :return:
    """
    code_length = tf.cast(tf.shape(tensor_in)[1], tf.float32)
    m1 = tensor_in - 1
    c1 = tf.matmul(tensor_in, m1, transpose_b=True)
    c2 = tf.matmul(m1, tensor_in, transpose_b=True)
    normalized_dist = tf.math.abs(c1 + c2) / code_length
    return tf.pow(1 - normalized_dist, 1.4)


@tf.function
def label_relevance(label):
    rel = tf.matmul(label, label, transpose_b=True)
    rel = tf.cast(tf.greater(rel, 0), tf.float32)
    return rel


@tf.custom_gradient
def ste(x):
    rslt = (tf.cast(tf.greater(x, .5), tf.float32) + 1.) / 2.

    def grad(d_x):
        return d_x

    return rslt, grad


def sigmoid_sign(logits, eps):
    """
    {0,1} sign function with (1) sigmoid activation (2) perturbation of eps in sigmoid
    :param logits: bottom layer output
    :param eps: randomly sampled values between [0,1]
    :return:
    """
    prob = 1.0 / (1 + tf.exp(-logits))
    code = (tf.sign(prob - eps) + 1.0) / 2.0
    return code, prob


@tf.custom_gradient
def binary_activation(logits, eps):
    """
    :param logits:
    :param eps:
    :return:
    """
    code, prob = sigmoid_sign(logits, eps)

    def grad(_d_code, _d_prob):
        """
        Distributional derivative with Bernoulli probs
        :param _d_code: bp gradients through code
        :param _d_prob: bp gradients through prob
        :return:
        """

        d_logits = prob * (1 - prob) * (_d_code + _d_prob)

        d_eps = _d_code
        return d_logits, d_eps

    return [code, prob], grad

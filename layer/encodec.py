from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from util.tf_helper import row_distance, row_distance_hamming


class BasicEncoder(tf.keras.layers.Layer):
    def __init__(self, out_size=(128, 128, 128), **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential()
        for i in range(out_size.__len__() - 1):
            self.net.add(tf.keras.layers.Dense(out_size[i], activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(out_size[-1]))

    def call(self, inputs, **kwargs):
        """

        :param inputs: x-[N dx], y-[N dy]
        :param kwargs:
        :return:
        """
        cat = tf.concat([inputs[0], inputs[1]], -1)

        return self.net(cat)


class BasicDecoder(tf.keras.layers.Layer):
    def __init__(self, out_size=(128, 128), **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential()
        for i in out_size:
            self.net.add(tf.keras.layers.Dense(i, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(2))

    def call(self, inputs, **kwargs):
        """

        :param inputs: context-[N D], targets-[N T d]
        :param kwargs:
        :return:
        """
        context = inputs[0]
        targets = inputs[1]
        seq_length = tf.shape(targets)[1]

        context = tf.expand_dims(context, 1)
        context = tf.tile(context, [1, seq_length, 1])

        cat = tf.concat([context, targets], -1)
        mean, log_var = tf.split(self.net(cat), num_or_size_splits=2, axis=-1)
        var = tf.nn.softplus(log_var) * .9 + .1
        return mean, var


class MultiContextEncoder(BasicEncoder):
    def call(self, inputs, **kwargs):
        return self.net(inputs)


class MultiContextDecoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim=512, code_length=32, **kwargs):
        super().__init__(**kwargs)
        self.middle_dim = middle_dim
        self.code_length = code_length
        self.fc_k = tf.keras.layers.Dense(middle_dim / 2)
        self.fc_base = tf.keras.layers.Dense(middle_dim, activation=tf.nn.relu)
        self.fc_q = tf.keras.layers.Dense(middle_dim / 2)
        self.fc_hash = tf.keras.layers.Dense(code_length, activation=tf.nn.sigmoid)

    # noinspection PyMethodOverriding
    def call(self, query, emb):
        fc_k = self.fc_k(emb)
        fc_base = self.fc_base(query)
        fc_q = self.fc_q(fc_base)
        fc_hash = self.fc_hash(fc_base)
        return fc_hash, fc_k, fc_q

    def loss(self, fc_hash, fc_k, fc_q):
        prod = tf.matmul(fc_q, fc_k, transpose_b=True) / tf.sqrt(self.middle_size)
        prod = tf.nn.softmax(prod, axis=1)  # [B k]
        sim = tf.exp(row_distance(prod, prod) / -0.01) * -1 + 1
        sim_hamming = row_distance_hamming(fc_hash)
        batch_size = tf.shape(fc_hash)[0]

        sim_loss = tf.reduce_mean(tf.nn.l2_loss(sim_hamming - sim)) / batch_size / batch_size

        quantized = tf.cast(tf.greater(fc_hash, .5), dtype=tf.float32)
        quantization_loss = tf.reduce_mean(tf.nn.l2_loss(quantized - fc_hash)) / batch_size / self.code_length

        return sim_loss, quantization_loss

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class BasicEncoder(tf.keras.layers.Layer):
    def __init__(self, out_size=(128, 128, 128), **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential()
        for i in out_size:
            self.net.add(tf.keras.layers.Dense(i, activation=tf.nn.relu))

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

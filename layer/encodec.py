from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from util.tf_helper import row_distance_cosine, row_distance_hamming, ste, binary_activation
import numpy as np


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
        self.fc_k = tf.keras.layers.Dense(middle_dim, use_bias=False)
        self.fc_v = tf.keras.layers.Dense(middle_dim, use_bias=False)
        self.fc_q = tf.keras.layers.Dense(middle_dim, use_bias=False)
        self.fc_hash = tf.keras.layers.Dense(code_length)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # noinspection PyMethodOverriding
    def call(self, feat, emb):
        fc_k = self.fc_k(emb)  # [k d]
        fc_v = self.fc_v(emb)  # [k d]
        fc_q = self.fc_q(feat)  # [N d]

        attended = self.dot_attention(fc_q, fc_k, fc_v) * .1 + fc_q
        attended = self.layernorm(attended)

        fc_hash = self.fc_hash(attended)
        return fc_hash

    def dot_attention(self, q, k, v):
        a = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / np.sqrt(self.middle_dim), axis=1)  # [N k]
        return a @ v

    def loss(self, fc_hash, vq_feat, step=-1):
        sim = tf.pow((row_distance_cosine(vq_feat, vq_feat) + 1) / 2, 2.5)
        eps = tf.ones_like(fc_hash, dtype=tf.float32) / 2.
        code, _ = binary_activation(fc_hash, eps)
        sim_hamming = row_distance_hamming(code)
        batch_size = tf.cast(tf.shape(fc_hash)[0], tf.float32)

        sim_loss = tf.reduce_mean(tf.nn.l2_loss(sim_hamming - sim)) / batch_size / batch_size

        quantized = (tf.cast(tf.greater(fc_hash, .0), dtype=tf.float32) + 1.) / 2.
        quantization_loss = 0 * tf.reduce_mean(tf.nn.l2_loss(quantized - fc_hash)) / batch_size / self.code_length

        if step >= 0:
            tf.summary.image('sim/code', tf.expand_dims(tf.expand_dims(sim_hamming, -1), 0), step=step, max_outputs=1)
            tf.summary.image('sim/sim', tf.expand_dims(tf.expand_dims(sim, -1), 0), step=step, max_outputs=1)
            tf.summary.scalar('loss_dec/sim_loss', sim_loss, step=step)
            tf.summary.scalar('loss_dec/q_loss', quantization_loss, step=step)
            tf.summary.scalar('code/ones', tf.reduce_mean(code), step=step)
        return sim_loss, quantization_loss


class DeterministicDecoder(MultiContextDecoder):
    def __init__(self, middle_dim=512, code_length=32, k=10, **kwargs):
        super().__init__(middle_dim=middle_dim, code_length=code_length, **kwargs)
        self.cls = tf.keras.layers.Dense(k)

    def call(self, feat, emb):
        fc_hash = super().call(feat, emb)

        eps = tf.ones_like(fc_hash, dtype=tf.float32) / 2.
        code, prob = binary_activation(fc_hash, eps)

        fc_cls = self.cls(code)

        return code, prob, fc_cls

    @staticmethod
    @tf.function
    def kld_loss(q: tf.Tensor, p=0.5):
        loss = q * tf.math.log(q) - q * tf.math.log(p) + (-q + 1) * tf.math.log(-q + 1) - (-q + 1) * tf.math.log(1 - p)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    # noinspection PyMethodOverriding
    def loss(self, code, prob, fc_cls, label, step=-1):
        sim_hamming = row_distance_hamming(code)
        cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, fc_cls))
        kld_loss = .1 * self.kld_loss(prob)

        if step >= 0:
            tf.summary.image('sim/code', tf.expand_dims(tf.expand_dims(sim_hamming, -1), 0), step=step, max_outputs=1)
            tf.summary.scalar('loss_dec/cls_loss', cls_loss, step=step)
            tf.summary.scalar('loss_dec/kld_loss', kld_loss, step=step)
            tf.summary.scalar('code/ones', tf.reduce_mean(code), step=step)

        return cls_loss, kld_loss

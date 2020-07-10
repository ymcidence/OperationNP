from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from util.tf_helper import row_distance


def nearest_context(feature, context):
    distances = row_distance(feature, context)
    min_ind = tf.cast(tf.argmin(distances, axis=1), dtype=tf.int32)
    min_ind = tf.expand_dims(min_ind, 1)
    rslt = tf.gather_nd(context, min_ind)
    return rslt, min_ind


@tf.custom_gradient
def vq(feature, context):
    _, min_ind = nearest_context(feature, context)

    def grad(d_feature, d_context):
        d_feature = d_feature + tf.gather_nd(d_context, min_ind)

        return d_feature, d_context

    return [feature, context], grad


class MultipleContext(tf.keras.layers.Layer):
    def __init__(self, k=20, emb_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.k = k
        self.emb = tf.Variable(initial_value=tf.random.normal([k, emb_dim], stddev=.01), trainable=True)

    def call(self, inputs, **kwargs):
        """
        :param inputs: features produced by the source encoder size of [B d2]
        :param kwargs:
        :return:
        """
        feature, emb = vq(inputs, self.emb)

        return feature, emb

    def loss(self, inputs, step=-1):
        """
        This is Loss1 in my doc
        :param inputs:  features produced by the source encoder size of [B d2]
        :param step:
        :return:
        """

        nearest = tf.stop_gradient(nearest_context(inputs, self.emb)[0])

        batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
        vq_loss = tf.reduce_mean(tf.nn.l2_loss(inputs - nearest)) / (self.emb_dim * 1.) / batch_size
        regu_loss = tf.reduce_mean(
            tf.nn.l2_loss(tf.matmul(self.emb, self.emb, transpose_b=True) - tf.eye(self.k))) / self.k / self.k

        if step >= 0:
            tf.summary.scalar('loss/regu', regu_loss, step=step)
            tf.summary.scalar('loss/vq', vq_loss, step=step)

        return vq_loss * 0 + regu_loss

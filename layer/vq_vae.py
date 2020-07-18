from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer import transformer
from util.tf_helper import row_distance_cosine as row_distance


def nearest_context(feature, context):
    distances = row_distance(feature, context) * -1
    min_ind = tf.cast(tf.argmin(distances, axis=1), dtype=tf.int32)
    k = tf.shape(context)[0]
    min_ind = tf.one_hot(min_ind, k, dtype=tf.float32)  # [N k]x`
    rslt = min_ind @ tf.stop_gradient(context)
    return rslt, min_ind


def lookup_table(ind, h, w):
    """
    This is identical to tf.one_hot. I'm just having a try.
    :param ind: [h 1] or [h]
    :param h: height or batch_size
    :param w: width or k in vq-vae
    :return: [h w] or [N k] in vq-vae
    """
    ind = tf.squeeze(ind)
    values = tf.ones_like(ind)
    col_ind = tf.expand_dims(ind, 1)
    row_ind = tf.expand_dims(tf.range(h, dtype=tf.int32), 1)
    rc_ind = tf.concat([row_ind, col_ind], 1)
    rc_ind = tf.cast(rc_ind, tf.int64)
    sparse = tf.SparseTensor(rc_ind, values, [h, w])

    return tf.stop_gradient(tf.cast(tf.sparse.to_dense(sparse), tf.float32))


@tf.custom_gradient
def vq(feature, context):
    value, ind = nearest_context(feature, context)

    def grad(d_value):
        d_context = tf.matmul(ind, d_value, transpose_a=True)
        return d_value, d_context

    return value, grad


class VQVAE(tf.keras.layers.Layer):
    def __init__(self, feat_dim=4096, middle_dim=512, k=20, emb_dim=256, top='relu', **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.middle_dim = middle_dim
        self.k = k
        self.emb_dim = emb_dim
        self.encoder = transformer.EncoderLayer(middle_dim, num_heads=4, dff=middle_dim, rate=0.)
        self.fc_1 = tf.keras.layers.Dense(emb_dim)

        self.decoder = transformer.EncoderLayer(middle_dim, num_heads=4, dff=middle_dim, rate=0.)
        self.fc_2 = tf.keras.layers.Dense(feat_dim, activation=top)

        self.emb = tf.Variable(initial_value=tf.random.normal([k, emb_dim], stddev=.01), trainable=True,
                               dtype=tf.float32)

    def call(self, inputs, **kwargs):
        encoded = self.encoder(tf.expand_dims(inputs, 0), training=True, mask=None)
        encoded = self.fc_1(tf.squeeze(encoded))
        vq_feat = vq(encoded, self.emb)
        _, context_ind = nearest_context(encoded, self.emb)
        # random = tf.random.normal(tf.shape(vq_feat), stddev=.05)
        # vq_feat = vq_feat + random

        decoded = self.decoder(tf.expand_dims(vq_feat, 0), training=True, mask=None)
        decoded = self.fc_2(tf.squeeze(decoded))
        return encoded, vq_feat, context_ind, decoded, self.emb

    def infer(self):
        decoded = self.decoder(tf.expand_dims(self.emb, 0))
        return self.fc_2(tf.squeeze(decoded))

    def loss(self, original, encoded, context_ind, decoded, beta=.25, step=-1):
        batch_size = tf.cast(tf.shape(original)[0], tf.float32)
        likelihood = tf.nn.l2_loss(original - decoded) / self.feat_dim / batch_size

        indexed_emb = context_ind @ self.emb
        kl_1 = tf.nn.l2_loss(tf.stop_gradient(encoded) - indexed_emb) / self.emb_dim / batch_size
        kl_2 = beta * tf.nn.l2_loss(tf.stop_gradient(indexed_emb) - encoded) / self.emb_dim / batch_size

        sim = (row_distance(self.emb, self.emb) + 1) / 2
        sim = tf.expand_dims(tf.expand_dims(sim, 0), -1)

        # orthogonality = tf.nn.l2_loss(tf.eye(self.k) - sim) / self.k / self.k

        loss = likelihood + kl_1 + kl_2

        if step >= 0:
            tf.summary.image('vq/emb', sim, step=step, max_outputs=1)
            tf.summary.scalar('loss_vq/likelihood', likelihood, step=step)
            tf.summary.scalar('loss_vq/kl_1', kl_1, step=step)
            tf.summary.scalar('loss_vq/kl_2', kl_2, step=step)
            tf.summary.scalar('loss_vq/loss', loss, step=step)
            # tf.summary.scalar('loss_vq/orthogonality', orthogonality, step=step)

        return loss


class MultiHeadVQVAE(VQVAE):
    def __init__(self, group=1, feat_dim=4096, middle_dim=512, k=20, emb_dim=256, top='relu', **kwargs):
        super().__init__(feat_dim=feat_dim, middle_dim=middle_dim, k=k, emb_dim=emb_dim, top=top, **kwargs)
        self.group = group
        self.grouped_emb = tf.split(self.emb, self.group, axis=1)

    def call(self, inputs, **kwargs):
        encoded = self.encoder(tf.expand_dims(inputs, 0), training=True, mask=None)
        encoded = self.fc_1(tf.squeeze(encoded))
        encoded = tf.split(encoded, self.group, axis=1)
        vq_tmp = []
        context_ind = []
        for i in range(self.group):
            vq_tmp.append(vq(encoded[i], self.grouped_emb[i]))
            context_ind.append(nearest_context(encoded[i], self.grouped_emb[i])[1])

        vq_feat = tf.concat(vq_tmp, axis=1)
        decoded = self.decoder(tf.expand_dims(vq_feat, 0), training=True, mask=None)
        decoded = self.fc_2(tf.squeeze(decoded))
        return encoded, vq_feat, context_ind, decoded, self.emb

    def loss(self, original, encoded, context_ind, decoded, beta=.25, step=-1):
        batch_size = tf.cast(tf.shape(original)[0], tf.float32)
        likelihood = tf.nn.l2_loss(original - decoded) / self.feat_dim / batch_size

        indexed_emb = []
        for i in range(self.group):
            indexed_emb.append(context_ind[i] @ self.grouped_emb[i])

        indexed_emb = tf.concat(indexed_emb, axis=1)

        _encoded = tf.concat(encoded, axis=1)
        kl_1 = tf.nn.l2_loss(tf.stop_gradient(_encoded) - indexed_emb) / self.emb_dim / batch_size
        kl_2 = beta * tf.nn.l2_loss(tf.stop_gradient(indexed_emb) - _encoded) / self.emb_dim / batch_size

        sim = (row_distance(self.emb, self.emb) + 1) / 2
        sim = tf.expand_dims(tf.expand_dims(sim, 0), -1)

        # orthogonality = tf.nn.l2_loss(tf.eye(self.k) - sim) / self.k / self.k

        loss = likelihood + kl_1 + kl_2

        if step >= 0:
            tf.summary.image('vq/emb', sim, step=step, max_outputs=1)
            tf.summary.scalar('loss_vq/likelihood', likelihood, step=step)
            tf.summary.scalar('loss_vq/kl_1', kl_1, step=step)
            tf.summary.scalar('loss_vq/kl_2', kl_2, step=step)
            tf.summary.scalar('loss_vq/loss', loss, step=step)
            # tf.summary.scalar('loss_vq/orthogonality', orthogonality, step=step)

        return loss


if __name__ == '__main__':
    a = tf.constant([[3], [4], [2], [2]], tf.int32)
    b = lookup_table(a, tf.cast(tf.shape(a)[0], tf.int64), 5)

    print(b)
    print(tf.one_hot(tf.squeeze(a), 5))

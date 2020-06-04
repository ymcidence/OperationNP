from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
from layer import encodec


class BasicCNP(tf.keras.Model):
    def __init__(self, sigmoid=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encodec.BasicEncoder()
        self.decoder = encodec.BasicDecoder()
        self.sigmoid = sigmoid

    # noinspection PyMethodOverriding
    def call(self, observations, targets):
        context = self.encoder(observations)  # [N T D]
        context = tf.reduce_mean(context, axis=1)  # [N D]
        mean, var = self.decoder([context, targets])
        if self.sigmoid:
            mean = tf.nn.sigmoid(mean)

        mvn = tfp.distributions.MultivariateNormalDiag(mean, var)
        return mean, var, mvn

    @staticmethod
    def obj(gt, pred: tfp.distributions.MultivariateNormalDiag):
        return -tf.reduce_mean(pred.log_prob(gt))


class MNISTCNP(BasicCNP):
    def __init__(self, sigmoid=False, img_size=28, emb_size=64, *args, **kwargs):
        super().__init__(sigmoid, *args, **kwargs)
        self.emb = tf.keras.layers.Embedding(img_size * img_size, emb_size)

    def call(self, observations, targets):
        o_emb = self.emb(tf.squeeze(observations[0]))

        context = self.encoder([o_emb, observations[1]])  # [N T D]
        context = tf.reduce_mean(context, axis=1)  # [N D]

        t_emb = self.emb(tf.squeeze(targets))
        mean, var = self.decoder([context, t_emb])
        if self.sigmoid:
            mean = tf.nn.sigmoid(mean)

        mvn = tfp.distributions.MultivariateNormalDiag(mean, var)
        return mean, var, mvn

    # @staticmethod
    # def obj(gt, pred):
    #     return tf.reduce_mean(tf.nn.l2_loss(gt - pred))

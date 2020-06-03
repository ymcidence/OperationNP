from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
from layer import encodec


class BasicCNP(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encodec.BasicEncoder()
        self.decoder = encodec.BasicDecoder()

    # noinspection PyMethodOverriding
    def call(self, observations, targets):
        context = self.encoder(observations)  # [N T D]
        context = tf.reduce_mean(context, axis=1)  # [N D]
        mean, var = self.decoder([context, targets])

        mvn = tfp.distributions.MultivariateNormalDiag(mean, var)
        return mean, var, mvn

    @staticmethod
    def obj(gt, pred: tfp.distributions.MultivariateNormalDiag):
        return -tf.reduce_mean(pred.log_prob(gt))

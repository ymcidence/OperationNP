from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from layer import encodec, vq_vae


class HashNP(tf.keras.Model):
    def __init__(self, feat_dim=4096, middle_dim=512, code_length=32, k=10, emb_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_dim = feat_dim

        self.encoder = vq_vae.VQVAE(feat_dim=feat_dim, k=k, emb_dim=emb_dim, top='relu')
        self.decoder = encodec.MultiContextDecoder(middle_dim=middle_dim, code_length=code_length)

    def call(self, inputs, training=None, mask=None):
        if training:
            encoded, vq_feat, context_ind, decoded, context = self.encoder(inputs)

            fc_hash = self.decoder(inputs, tf.stop_gradient(context))

            rslt = dict()
            rslt['encoder'] = [encoded, vq_feat, context_ind, decoded, context]
            rslt['decoder'] = fc_hash

            return rslt
        else:
            return (tf.cast(tf.greater(self.decoder(inputs, self.encoder.emb), .0), tf.float32) + 1) / 2

    def loss(self, inputs, net_out, lamb=(1, 1, 1), step=-1):
        encoded, vq_feat, context_ind, decoded, _ = net_out['encoder']
        fc_hash = net_out['decoder']

        l1 = self.encoder.loss(inputs, encoded, context_ind, decoded, step=step)

        l2, l3 = self.decoder.loss(fc_hash, tf.stop_gradient(vq_feat), step=step)
        rslt = lamb[0] * l1 + lamb[1] * l2 + lamb[2] * l3

        if step >= 0:
            tf.summary.scalar('loss/loss', rslt, step=step)

        return rslt


class DetHashNP(tf.keras.Model):
    def __init__(self, feat_dim=4096, middle_dim=512, code_length=32, k=10, emb_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_dim = feat_dim

        self.encoder = vq_vae.VQVAE(feat_dim=feat_dim, k=k, emb_dim=emb_dim, top='relu')
        self.decoder = encodec.DeterministicDecoder(middle_dim=middle_dim, code_length=code_length, k=k)

    def call(self, inputs, training=None, mask=None):
        if training:
            encoded, vq_feat, context_ind, decoded, context = self.encoder(inputs)

            code, prob, fc_cls = self.decoder(inputs, context)

            rslt = dict()
            rslt['encoder'] = [encoded, vq_feat, context_ind, decoded, context]
            rslt['decoder'] = [code, prob, fc_cls]

            return rslt
        else:
            return self.decoder(inputs, self.encoder.emb)[0]

    def loss(self, inputs, net_out, lamb=(1, 1, 1), step=-1):
        encoded, vq_feat, context_ind, decoded, _ = net_out['encoder']
        code, prob, fc_cls = net_out['decoder']

        l1 = self.encoder.loss(inputs, encoded, context_ind, decoded, step=step)

        l2, l3 = self.decoder.loss(code, prob, fc_cls, context_ind, step=step)
        rslt = lamb[0] * l1 + lamb[1] * l2 + lamb[2] * l3

        if step >= 0:
            tf.summary.scalar('loss/loss', rslt, step=step)

        return rslt

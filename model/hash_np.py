from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from layer import encodec, k_context


class HashNP(tf.keras.Model):
    def __init__(self, middle_dim=512, code_length=64, k=20, emb_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = encodec.MultiContextEncoder(out_size=(middle_dim, emb_dim))
        self.decoder = encodec.MultiContextDecoder(middle_dim=middle_dim, code_length=code_length)
        self.context = k_context.MultipleContext(k=k, emb_dim=emb_dim)

    def call(self, inputs, training=None, mask=None):
        if training:
            source_feat = self.encoder(inputs)
            source_feat, context = self.context(source_feat)
            fc_hash, fc_k, fc_q = self.decoder(inputs, context)
            return source_feat, context, fc_hash, fc_k, fc_q
        else:
            return self.decoder(inputs, self.context.emb)

    def loss(self, source_feat, fc_hash, fc_k, fc_q, lamb=(1, 1, 1)):

        l1 = self.context.loss(source_feat)
        l2, l3 = self.decoder.loss(fc_hash, fc_k, fc_q)
        return lamb[0] * l1 + lamb[1] * l2 + lamb[2] * l3

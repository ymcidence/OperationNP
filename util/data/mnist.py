from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from util.plot import gen_indices


class MNISTData(object):
    def __init__(self, batch_size=64, max_ob=256):
        data = tf.keras.datasets.mnist.load_data()
        (x_train, self.y_train), (x_test, self.y_test) = data
        x_train, x_test = x_train / 255., x_test / 255.

        shape = x_train.shape
        self.x_train = np.reshape(x_train, [shape[0], shape[1] * shape[2]])
        shape = x_test.shape
        self.x_test = np.reshape(x_test, [shape[0], shape[1] * shape[2]])
        self.batch_size = batch_size
        self.dim = shape[1] * shape[2]
        self.max_ob = max_ob
        self._build()

    def _build(self):
        def parser(d):
            _x = tf.cast(d['x'], tf.float32)

            num_ob = tf.random.uniform(shape=(), minval=64, maxval=self.max_ob, dtype=tf.int32)

            ob_pos = tf.random.uniform([self.batch_size, num_ob], 0, self.dim - 1, dtype=tf.int32)
            ob_pos = tf.expand_dims(ob_pos, -1)

            indices = gen_indices(ob_pos)
            ob_value = tf.gather_nd(_x, indices)
            ob_pos = (tf.cast(ob_pos, tf.float32) * 2. / self.dim) - 1

            tar_pos = tf.range(-1, 1, 2 / self.dim, dtype=tf.float32)
            tar_pos = tf.tile(tf.expand_dims(tar_pos, axis=0), [self.batch_size, 1])
            tar_value = _x
            rslt = [ob_value, tar_pos, tar_value]
            return [ob_pos] + [tf.expand_dims(i, -1) for i in rslt]

        data_train = tf.data.Dataset.from_tensor_slices({'x': self.x_train, 'l': self.y_train})
        data_test = tf.data.Dataset.from_tensor_slices({'x': self.x_test, 'l': self.y_test})

        data_train = data_train.cache().repeat().shuffle(1000).batch(self.batch_size)
        data_test = data_test.cache().repeat().shuffle(1000).batch(self.batch_size)

        self.data_train = data_train.map(parser)
        self.data_test = data_test.map(parser)


if __name__ == '__main__':
    mnist = MNISTData()
    a = mnist.data_train
    b = mnist.data_test

    ia = iter(a)
    print(next(ia))

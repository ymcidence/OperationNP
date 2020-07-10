from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os

from meta import ROOT_PATH
from time import gmtime, strftime
from layer.vq_vae import VQVAE as Model
from util.data.mnist import MNISTData as Data


def plot_mnist_gen(x, k=10, size=28):
    target_shape = [k, size, size, 1]
    _x = tf.reshape(x, target_shape)
    _x = tf.split(_x, k, axis=0)
    return tf.concat(_x, axis=2)


def step_train(model: Model, data, opt: tf.optimizers.Optimizer, t):
    batch_data = next(data)
    x = tf.cast(batch_data['x'], tf.float32)
    summary_step = -1 if t % 50 > 0 else t
    with tf.GradientTape() as tape:
        encoded, vq_feat, context_ind, decoded, _ = model(x)
        loss = model.loss(x, encoded, context_ind, decoded, step=summary_step)
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))

    if summary_step >= 0:
        gt = tf.reshape(x, [-1, 28, 28, 1])
        rec = tf.reshape(decoded, [-1, 28, 28, 1])
        tf.summary.image('train/gt', gt, summary_step, max_outputs=1)
        tf.summary.image('train/rec', rec, summary_step, max_outputs=1)

    return loss.numpy()


def hook(model: Model, data, t):
    img = model.infer()
    img = plot_mnist_gen(img)
    batch_data = next(data)
    x = tf.cast(batch_data['x'], tf.float32)

    encoded, vq_feat, context_ind, decoded, _ = model(x)
    loss = model.loss(x, encoded, context_ind, decoded)
    tf.summary.scalar('loss/test', loss, step=t)
    tf.summary.image('test/mean', img, t, max_outputs=1)
    return loss.numpy()


model = Model(feat_dim=784, k=10, top='sigmoid')
data = Data(max_ob=-1)
train_data = iter(data.data_train)
test_data = iter(data.data_test)
opt = tf.keras.optimizers.Adam(1e-4)

time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
result_path = os.path.join(ROOT_PATH, 'result', 'MNIST')
save_path = os.path.join(result_path, 'model', time_string)
summary_path = os.path.join(result_path, 'log', time_string)
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

writer = tf.summary.create_file_writer(summary_path)
checkpoint = tf.train.Checkpoint(opt=opt, model=model)

for i in range(10000):
    with writer.as_default():
        train_loss = step_train(model, train_data, opt, i)
        print('Step: {}, Loss: {}'.format(i, train_loss))
        if (i + 1) % 200 == 0:
            print('-----------Testing-----------')
            test_loss = hook(model, test_data, i)
            print('Hook: {}, Loss: {}'.format(i, test_loss))

            save_name = os.path.join(save_path, 'ym' + str(i))
            checkpoint.save(file_prefix=save_name)

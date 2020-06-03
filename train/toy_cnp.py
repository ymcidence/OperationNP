from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf

from meta import ROOT_PATH
from time import gmtime, strftime
from model.cnp import BasicCNP as Model
from util.data.toy_data import GPData as Data
from util.plot import plot_toy

MAX_ITER = 10000


def step_train(model: Model, data: Data, opt: tf.optimizers.Optimizer, t):
    o_x, o_y, t_x, t_y = data.batch()
    with tf.GradientTape() as tape:
        mean, var, mvn = model([o_x, o_y], t_x)
        loss = model.obj(t_y, mvn)
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))
        tf.summary.scalar('train/loss', loss.numpy(), step=t)

    return loss.numpy()


def hook(model: Model, data: Data, t):
    o_x, o_y, t_x, t_y = data.batch()
    mean, var, mvn = model([o_x, o_y], t_x)
    loss = model.obj(t_y, mvn)
    img = plot_toy(t_x, t_y, o_x, o_y, mean, var, True)
    tf.summary.scalar('test/loss', loss.numpy(), t)
    tf.summary.image('test/plot', img, t, max_outputs=1)
    return loss.numpy()


model = Model()
train_data = Data(64, 10)
test_data = Data(64, 10, test=True)
opt = tf.keras.optimizers.Adam(1e-3)

time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
result_path = os.path.join(ROOT_PATH, 'result', 'toy')
save_path = os.path.join(result_path, 'model', time_string)
summary_path = os.path.join(result_path, 'log', time_string)
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

writer = tf.summary.create_file_writer(summary_path)
checkpoint = tf.train.Checkpoint(opt=opt, model=model)

for i in range(MAX_ITER):
    with writer.as_default():
        train_loss = step_train(model, train_data, opt, i)
        print('Step: {}, Loss: {}'.format(i, train_loss))
        if i == 0:
            print(model.summary())
        if (i + 1) % 200 == 0:
            print('-----------Testing-----------')
            test_loss = hook(model, test_data, i)
            print('Hook: {}, Loss: {}'.format(i, test_loss))

            save_name = os.path.join(save_path, 'ym' + str(i))
            checkpoint.save(file_prefix=save_name)

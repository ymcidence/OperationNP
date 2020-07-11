from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from time import gmtime, strftime
from util.data.cifar import Dataset
from util.tf_helper import label_relevance
from util.eval_tools import eval_cls_map
from model.hash_np import DetHashNP as Model
from meta import ROOT_PATH


def train_step(model: Model, batch_data, opt: tf.optimizers.Optimizer, step):
    feat = batch_data[1]
    label = batch_data[2]
    summary_step = -1 if step % 50 > 0 else step
    with tf.GradientTape() as tape:
        net_out = model(feat, training=True)
        loss = model.loss(feat, net_out, step=summary_step)
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))

    code, prob, fc_cls = net_out['decoder']
    if summary_step >= 0:
        sim_gt = tf.expand_dims(tf.expand_dims(label_relevance(label), 0), -1)
        batch_map = eval_cls_map(code.numpy(), code.numpy(), label.numpy(), label.numpy())
        tf.summary.image('sim/gt', sim_gt, step=summary_step, max_outputs=1)
        tf.summary.scalar('map/train', batch_map, step=summary_step)

    return code, loss


def train_step_det(model: Model, batch_data, opt: tf.optimizers.Optimizer, step):
    feat = batch_data[1]
    label = batch_data[2]
    summary_step = -1 if step % 50 > 0 else step
    with tf.GradientTape() as tape:
        net_out = model(feat, training=True)
        loss = model.loss(feat, net_out, step=summary_step)
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))

    fc_hash = net_out['decoder']
    code = (tf.cast(tf.greater(fc_hash, .0), tf.float32) + 1) / 2
    if summary_step >= 0:
        sim_gt = tf.expand_dims(tf.expand_dims(label_relevance(label), 0), -1)
        batch_map = eval_cls_map(code.numpy(), code.numpy(), label.numpy(), label.numpy())
        tf.summary.image('sim/gt', sim_gt, step=summary_step, max_outputs=1)
        tf.summary.scalar('map/train', batch_map, step=summary_step)

    return code, loss


def train(max_iter=10000):
    model = Model()
    data = Dataset()
    opt = tf.keras.optimizers.Adam(1e-4)
    train_iter = iter(data.train_data)
    test_iter = iter(data.test_data)

    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    result_path = os.path.join(ROOT_PATH, 'result', 'cifar10')
    save_path = os.path.join(result_path, 'model', time_string)
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(opt=opt, model=model)

    for i in range(max_iter):
        with writer.as_default():
            train_batch = next(train_iter)

            train_code, train_loss = train_step(model, train_batch, opt, i)
            train_entry = train_batch[0]
            train_label = train_batch[2]
            data.update(train_entry.numpy(), train_code.numpy(), train_label.numpy(), 'train')
            print('Step {}: loss: {}'.format(i, train_loss.numpy()))

            if i % 500 == 0 and i > 0:
                print('Testing!!!!!!!!')
                test_batch = next(test_iter)
                test_entry, test_feat, test_label = test_batch
                test_code = model(test_feat, training=False)
                data.update(test_entry.numpy(), test_code.numpy(), test_label.numpy(), 'test')
                test_map = eval_cls_map(test_code.numpy(), data.train_code, test_label.numpy(), data.train_label,
                                        at=1000)

                tf.summary.scalar('map/test', test_map, step=i)


if __name__ == '__main__':
    train()

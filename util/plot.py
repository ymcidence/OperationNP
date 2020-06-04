import io
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_toy(target_x, target_y, context_x, context_y, pred_y, var, tb=False):
    # Plot everything
    p = plt.figure()
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    if tb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(p)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image
    else:
        plt.show()


def gen_indices(col_ind):
    """

    :param col_ind: [N D 1]
    :return:
    """
    shape = tf.shape(col_ind)
    row_ind = tf.range(0, shape[0])
    row_ind = tf.expand_dims(tf.tile(tf.expand_dims(row_ind, -1), [1, shape[1]]), -1)
    ind = tf.concat([row_ind, col_ind], axis=-1)
    return ind


def plot_mnist(o_x, o_y, t_y, gt_y, size=28):
    batch_size = tf.shape(o_x)[0]
    target_shape = [batch_size, size, size, 1]
    original_shape = [batch_size, size * size]

    # _o_x = tf.cast((o_x + 1) / 2 * (size * size), tf.int32)
    _o_x = tf.cast(o_x, tf.int32)
    indices = gen_indices(_o_x)
    _o_y = tf.scatter_nd(indices, tf.squeeze(o_y), original_shape)

    p_1 = tf.reshape(_o_y, target_shape)
    p_2 = tf.reshape(gt_y, target_shape)
    p_3 = tf.reshape(t_y, target_shape)

    return tf.concat([p_1, p_2, p_3], axis=2)
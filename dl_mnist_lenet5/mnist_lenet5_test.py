import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import dl_mnist_lenet5.mnist_lenet5_forward as forward
import dl_mnist_lenet5.mnist_lenet5_backward as backword
import numpy as np

TEST_INTERVAL_SECS = 5


def t(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,
            forward.IMAGE_SIZE,
            forward.IMAGE_SIZE,
            forward.NUM_CHANNELS
        ])

        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(backword.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backword.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    globals_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(mnist.test.images, (
                        mnist.test.num_examples,
                        forward.IMAGE_SIZE,
                        forward.IMAGE_SIZE,
                        forward.NUM_CHANNELS
                    ))
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, y_: mnist.test.labels})
                    print('After %s training step(s), test accuracy = %g' % (globals_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    t(mnist)


if __name__ == '__main__':
    main()

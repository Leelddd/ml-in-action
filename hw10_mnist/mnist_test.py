import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"


def t(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # saver = tf.train.Saver()
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:

                for path in ckpt.all_model_checkpoint_paths:
                    saver.restore(sess, path)

                    globals_step = path.split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print('After %s training step(s), test accuracy = %g' % (globals_step, accuracy_score))


def main(argv=None):
    mnist = input_data.read_data_sets("data/", one_hot=True)
    t(mnist)


if __name__ == '__main__':
    tf.app.run()

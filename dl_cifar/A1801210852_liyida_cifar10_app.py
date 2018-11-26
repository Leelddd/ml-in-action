import tensorflow as tf
import numpy as np
from PIL import Image
# import dl.course.A1801210852_liyida_mnist_backward as mnist_backward
# import dl.course.A1801210852_liyida_mnist_forward as mnist_forward
import A1801210852_liyida_cifar10_backward as mnist_backward
import A1801210852_liyida_cifar10_forward as mnist_forward


def restore_model(testPicArr):  # restore model
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])  # init x by input node number
        y = mnist_forward.forward(x, None)  # init y by forward
        preValue = tf.argmax(y, 1)  # get max index of y

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)  # ema
        # init saver which could restore ema
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)  # get checkpoint by path
            if ckpt and ckpt.model_checkpoint_path:  # if checkpoint exist
                saver.restore(sess, ckpt.model_checkpoint_path)  # restore model
                preValue = sess.run(preValue, feed_dict={x: testPicArr})  # predict value
                return preValue  # return predict value
            else:  # if no check point, return -1
                print("No check point file found")
                return -1


def pre_pic(picName):  # preprocess picture
    img = Image.open(picName)  # open image by path
    reIm = img.resize((32, 32), Image.ANTIALIAS)  # resize picture to 28x28
    im_arr = np.array(reIm.convert('L'))  # get gray level array
    threshold = 50  # < threshold to 0, > threshold to 255
    for i in range(32):
        for j in range(32):
            # set value by threshold
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 1024])  # reshape image arr to 1,784
    nm_arr = nm_arr.astype(np.float32)  # set type as float32
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)  # set 255 to 1

    return img_ready


def application():
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    testNum = input("input the number of test pictures:")  # read picture number
    for i in range(int(testNum)):
        testPic = input("the path of test picture:")  # read one picture path
        testPicArr = pre_pic(testPic)  # preprocess picture
        preValue = restore_model(testPicArr)  # predict value
        # print("The prediction number is:", preValue)  # print predict value
        print("The prediction type is:", labels[int(preValue)])  # print predict value


def _application():
    l = [3, 10, 21]
    for i in range(3):
        testPicArr = pre_pic(
            'C:/Users/Leeld/Documents/projects/ml-in-action/dl/cifar/cifar-10/test/airplane/batch_1_num_' + str(
                l[i]) + '.jpg')
        preValue = restore_model(testPicArr)
        print(i)
        print("The prediction number is:", preValue)


def main():
    application()
    # _application()


if __name__ == '__main__':
    main()

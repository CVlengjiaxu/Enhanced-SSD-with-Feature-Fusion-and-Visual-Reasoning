import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def tensor4_resize(input_tensor, output_size):
    # resize each image of input_tensor
    resized_tensor = []
    with tf.Session() as sess:
        input_shape = tf.shape(input_tensor)
        batch_num = input_shape[0].eval()
        channel_num = input_shape[3].eval()
        for i in range(batch_num):
            resized = tf.image.resize_images(input_tensor[i], output_size, method=0)
            resized = tf.reshape(resized, shape=[1, output_size[0], output_size[1], channel_num])
            if i == 0:
                resized_tensor = resized
            else:
                resized_tensor = tf.concat([resized_tensor,resized], 0)
    return resized_tensor

if __name__ == '__main__':
    # input_tensor = tf.random_normal([4,200,200,1])
    #
    # img = plt.imread('../demo/000001.jpg')
    # plt.figure(0)
    # plt.imshow(img)
    # plt.show()
    #
    img = tf.read_file('../demo/000001.jpg')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, shape=[1,500,353,3])
    input_tensor = img

    output_size = [2000,1000]
    resized_tensor = tensor4_resize(input_tensor, output_size)
    with tf.Session() as sess:
        input_tensor_r, resized_tensor_r = sess.run([input_tensor, resized_tensor])
        plt.figure(2)
        input_tensor_r = np.asanyarray(input_tensor_r, 'uint8')
        plt.imshow(input_tensor_r[0,:,:,1])
        plt.figure(3)
        resized_tensor_r = np.asanyarray(resized_tensor_r, 'uint8')
        plt.imshow(resized_tensor_r[0,:,:,1])
        plt.show()


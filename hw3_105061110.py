
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_prepro import data_preprocess, label_generator


class CNN():
    def __init__(self, imsize, X_train, y_train, X_test, y_test):
        self.height = imsize[0]
        self.width = imsize[1]
        self.channel = 1
        if len(imsize) > 2:
            self.channel = imsize[2]
        self.classes = 101


        self.xs = tf.placeholder(tf.float32, [None, self.height, self.width])/255
        self.ys = tf.placeholder(tf.float32, [None, self.classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.x_image = tf.reshape(self.xs, [-1, self.height, self.width, self.channel])
        # print(self.x_image.shape)

        ## conv1
        # kernel = 5x5, channel= 1, feature_maps = 8
        self.fmaps_conv1 = 16
        self.ker_conv1 = 3
        self.W_conv1 = self.init_weights([self.ker_conv1, self.ker_conv1, self.channel, self.fmaps_conv1])
        self.b_conv1 = self.init_biases([self.fmaps_conv1])
        self.conv1 = self.conv2d(self.x_image, self.W_conv1) + self.b_conv1
        self.a_conv1 = tf.nn.relu(self.conv1)               # output size 128x128x16
        self.p_conv1 = self.max_pool(self.a_conv1, rate=2)  # output size 64x64x16

        ## conv2
        # kernel = 5x5, channel= 32, feature_maps = 32
        self.fmaps_conv2 = 32
        self.ker_conv2 = 3
        self.W_conv2 = self.init_weights([self.ker_conv2, self.ker_conv2, self.fmaps_conv1, self.fmaps_conv2])
        self.b_conv2 = self.init_biases([self.fmaps_conv2])
        self.conv2 = self.conv2d(self.p_conv1, self.W_conv2) + self.b_conv2
        self.a_conv2 = tf.nn.relu(self.conv2)               # output size 64x64x16
        self.p_conv2 = self.max_pool(self.a_conv2, rate=4)  # output size 16x16x16

        ## fcn1
        self.size_conv2 = 16
        self.hidden_fcn1 = 1024
        self.W_fcn1 = self.init_weights([self.size_conv2*self.size_conv2*self.fmaps_conv2, self.hidden_fcn1])
        self.b_fcn1 = self.init_biases([self.hidden_fcn1])
        self.f_fcn1 = tf.reshape(self.p_conv2, [-1, self.size_conv2*self.size_conv2*self.fmaps_conv2]) # flatten
        self.a_fcn1 = tf.nn.relu(tf.matmul(self.f_fcn1, self.W_fcn1) + self.b_fcn1)
        # self.a_fcn1 = tf.nn.tanh(tf.matmul(self.f_fcn1, self.W_fcn1) + self.b_fcn1)
        self.d_fcn1 = tf.nn.dropout(self.a_fcn1, self.keep_prob)

        ## fcn2
        self.W_fcn2 = self.init_weights([self.hidden_fcn1, self.classes])
        self.b_fcn2 = self.init_biases([self.classes])
        self.prediction = tf.nn.softmax(tf.matmul(self.d_fcn1, self.W_fcn2) + self.b_fcn2)

        ## loss fucniton: cross entropy
        # self.loss =  tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction), reduction_indices=[1]))
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.ys, logits=self.prediction)
        ## optimizer: Adam
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        ## session setup
        self.init = tf.global_variables_initializer()
        # self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.sess = tf.Session()
        self.sess.run(self.init)
        ## Training session
        # for i in range(1000):
        #     batch_xs, batch_ys = mnist.train.next_batch(100)
        #     self.sess.run(self.train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: 0.5})
        #     if i % 50 == 0:
        #         print(self.compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        #
        self.epochs = 100
        self.batch_size = 32
        loss = np.zeros((self.epochs, 1))
        accuracy = np.zeros((self.epochs, 1))
        for i in range(self.epochs):
            X_batch, y_batch = next_batch(self.batch_size, X_train, y_train)
            self.sess.run(self.train_step,
                feed_dict={self.xs:X_batch, self.ys:y_batch, self.keep_prob:1})

            if i % 10 == 0:
                loss[i] = np.mean(self.sess.run(self.loss, feed_dict={self.xs:X_batch, self.ys:y_batch, self.keep_prob:1}))
                print("loss: {}".format(loss[i][0]))
                print("acc: {}".format(self.compute_accuracy(X_test, y_test)))
        
        self.sess.close()
    
    def init_weights(self, shape):
        init = tf.truncated_normal(shape, stddev=.1)
        return tf.Variable(init)

    def init_biases(self, shape):
        init = tf.constant(.1, shape=shape)
        return tf.Variable(init)

    def conv2d(self, x, W):
        # strides = [1, x_step, y_step, 1], zero-padding
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(self, x, rate=2):
        # ->kernel = 4x4, strides = (4,4) 
        return tf.nn.max_pool(x, ksize=[1, rate, rate, 1], strides=[1, rate, rate, 1], padding='SAME')

    
    def evaluate(self):
        pass
    def predict(self):
        pass
    def compute_accuracy(self, v_xs, v_ys):
        # global prediction
        y_pre = self.sess.run(self.prediction, feed_dict={self.xs: v_xs, self.keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.xs: v_xs, self.ys: v_ys, self.keep_prob: 1})
        return result

    def __del__(self):
        self.sess.close()

def next_batch(batch_size, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
        
if __name__ == "__main__":
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # cnn = CNN((28, 28), mnist)
    
    X_train, name_train, y_train = label_generator("prepro_train")
    X_test, name_test, y_test = label_generator("prepro_test") # narray
    cnn = CNN(imsize=(128, 128), 
            X_train=X_train, 
            y_train=y_train.astype(np.int32),
            X_test=X_test,
            y_test=y_test.astype(np.int32)
            )

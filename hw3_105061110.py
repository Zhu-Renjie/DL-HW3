#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_prepro import data_prepro

class CNN():
    def __init__(self, imsize):
        self.height = imsize[0]
        self.width = imsize[1]
        self.channel = 1
        if len(imsize) > 2:
            self.channel = imsize[2]
        self.classes = 101

        self.xs = tf.placeholder(tf.float32, [None, self.height*self.width])/255
        self.ys = tf.placeholder(tf.float32, [None, self.classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.x_image = tf.reshape(self.xs, [-1, self.height, self.width, self.channel])
        # print(x_image.shape)

        ## conv1
        # kernel = 5x5, channel= 1, feature_maps = 32
        self.fmaps_conv1 = 32
        self.W_conv1 = self.init_weights([5, 5, self.channel, self.fmaps_conv1])
        self.b_conv1 = self.init_biases([self.fmaps_conv1])
        self.conv1 = self.conv2d(self.x_image, self.W_conv1) + self.b_conv1
        self.a_conv1 = tf.nn.relu(self.conv1) # output size 256x256x32
        self.p_conv1 = self.max_pool(self.a_conv1)  # output size 64x64x32

        ## conv2
        # kernel = 5x5, channel= 32, feature_maps = 64
        self.fmaps_conv2 = 64
        self.W_conv2 = self.init_weights([5, 5, self.fmaps_conv1, self.fmaps_conv2])
        self.b_conv2 = self.init_biases([self.fmaps_conv2])
        self.conv2 = self.conv2d(self.p_conv1, self.W_conv2) + self.b_conv2
        self.a_conv2 = tf.nn.relu(self.conv2) # output size 64x64x64
        self.p_conv2 = self.max_pool(self.a_conv2)  # output size 16x16x64

        ## fcn1
        self.hidden_fcn1 = 1024
        self.W_fcn1 = self.init_weights([16*16*64, self.hidden_fcn1])
        self.b_fcn1 = self.init_biases([self.hidden_fcn1])
        self.f_fcn1 = tf.reshape(self.p_conv2, [-1, 16*16*64]) # flatten
        self.a_fcn1 = tf.nn.relu(tf.matmul(self.f_fcn1, self.W_fcn1) + self.b_fcn1)
        self.d_fcn1 = tf.nn.dropout(self.a_fcn1, self.keep_prob)

        ## fcn2
        self.W_fcn2 = self.init_weights([self.hidden_fcn1, self.classes])
        self.b_fcn2 = self.init_biases([self.classes])
        self.prediction = tf.nn.softmax(tf.matmul(self.d_fcn1, self.W_fcn2) + self.b_fcn2)

        ## loss fucniton: cross entropy
        self.loss =  tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction), reduction_indices=[1]))
        ## optimizer: Adam
        self.train_step = tf.train.AdamOptimizer(3e-4).minimize(self.loss)

        ## session setup
        self.init = tf.global_variables_initializer()
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.sess.run(self.init)
        ## Training session

        ##
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

    def max_pool(self, x, rate=4):
        # ->kernel = 4x4, strides = (4,4) 
        return tf.nn.max_pool(x, ksize=[1, rate, rate, 1], strides=[1, rate, rate, 1], padding='SAME')
    
    def train(self):
        pass
    def evaluate(self):
        pass
    def predict(self):
        pass
    def get_metrics(self):
        pass

if __name__ == "__main__":
    cnn = CNN((256, 256))
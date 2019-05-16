
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_prepro import data_preprocess, label_generator
import os


class CNN():
    def __init__(self):
        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            # self.conv_layer(x, name='convtest', kernel_size=(3,3), n_output_channels=32)
            self.fc_layer(x, name='fctest', n_output_units=32, activation_fn=tf.nn.relu)

        del g, x

    def build_cnn():
        tf_x = tf.placeholder(tf.float32, shape=[None, 784], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

        tf_x_image = tf.reshape(tf.x, shape=[-1, 28, 28, 1], name='tf_x_reshape')
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name='tf_y_onehot')
        print('\nBuild 1st layer')
        h1 = self.conv_layer(tf.tf_x_image, name='conv_1', kernel_size=(5,5), 
                            padding_mode='VALID', n_output_channels=32)
        h1_pool = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        print('\nBuild 2nd layer')
        h2 = self.conv_layer(h1_pool, name='conv_2', kernel_size=(5,5), padding_mode='VALID',
                            n_output_channels=64)
        h2_pool = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        print('\nBuild 3rd layer')
        h3 = self.fc_layer(h2_pool, name='fc_3', n_output_units=1024,
                         activation_fn=tf.nn.relu)
        keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
        h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')

        print('\nBuild 4th layer')
        h4 = self.fc_layer(h3_drop, name='fc_4', n_output_units=10, activation_fn=None)

        predictions = {
            'probabilities' : tf.nn.softmax(h4, name='probabilities'),
            'labels' : tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
        }

        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=h4, labels=tf_y_onehot
            )
        )

        learning_rate = 1e-4
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

        correct_predictions = tf.equal(
            predictions=['labels'],
            tf_y, name='correct_preds'
        )

        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy'
        )

    def save(self, saver, sess, epoch, path='model/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        print('Saving model in {}'.format(path))
        saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step=epoch)

    def load(self, saver, sess, path, epoch):
        print('Loading model from {}'.format(path))
        saver.restore(sess, os.path.join(path, 'cnn-model.ckpt-{}'.format(epoch)))


    def train(self, sess, training_set, validation_set=None, initialize=True, epochs=20, shuffle=True,
            dropout=0.5, random_seed=None):
            
            X_data = np.array(training_set[0])
            y_data = np.array(training_set[1])
            training_loss= []

            if initialize:
                sess.run(tf.global_variables_initializer())

            np.random.seed(random_seed)
            for epoch in range(1, epochs+1):
                batch_gen = batch_generator(X_data, y_data, shuffle=shuffle)
            
            avg_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {
                    'tf_x:0' : batch_x,
                    'tf_y_0' : batch_y,
                    'fc_keep_prob:0' : dropout
                }
                loss, _ = sess.run(
                    ['cross_entropy_loss:0', 'train_op'],
                    feed_dict = feed
                )
                avg_loss += loss
            training_loss.append(avg_loss / (i+1))
            print("Epoch {} Training Avg. Loss: {}".format(epoch, avg_loss), end=' ')

            if validation_set is not None:
                feed = {
                    'tf_x:0' : validation_set[0],
                    'tf_y:0' : validation_set[1],
                    'fc_keep_prob:0' : 1.0
                }
                valid_acc = sess.run('accuracy:0', feed_dict=feed)
                print(" Validation Acc: {}".format(valid_acc))
            else:
                print()

    def conv_layer(self, input_tensor, name, kernel_size, n_output_channels, padding_mode='SAME', strides=(1,1,1,1)):
        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()
            n_input_channels = input_shape[-1]

            weights_shape = (list(kernel_size) + [n_input_channels, n_output_channels])
            weights = tf.get_variable(name='_weights', shape=weights_shape)
            print(weights)

            biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
            print(biases)

            conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
            print(conv)
            conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
            print(conv)
            conv = tf.nn.relu(conv, name='activation')
            
            return conv

    def fc_layer(self, input_tensor, name, n_output_units, activation_fn=None):
        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()[1:]
            n_input_units = np.prod(input_shape)
            if len(input_shape) > 1:
                input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))
            
            weight_shape = [n_input_units, n_output_units]
            weights = tf.get_variable(name='_weights', shape=weight_shape)
            print(weights)

            biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_units]))
            print(biases)

            layer = tf.matmul(input_tensor, weights)
            print(layer)
            layer = tf.nn.bias_add(layer, biases, name='net_pre-activation')
            print(layer)

            if activation_fn is None:
                return layer

            layer = activation_fn(layer)
            print(layer)
            return layer
    
def next_batch(batch_size, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
        
if __name__ == "__main__":
    # X_train, name_train, y_train = label_generator("prepro_train")
    # X_test, name_test, y_test = label_generator("prepro_test") # narray
    cnn = CNN()
    pass

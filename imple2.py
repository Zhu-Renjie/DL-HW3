
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_prepro import data_preprocess, label_generator
import os


class CNN():
    def __init__(self, batchsize=64, epochs=10, learning_rate=1e-4,
                dropout_rate=0.5, shuffle=True, random_seed=None):
        
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs= epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.training_loss = []
        self.training_accuracy = []
        self.testing_loss = []
        self.testing_accuracy = []

        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(random_seed)
            self.build_cnn()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=g)

    def build_cnn(self):
        tf_x = tf.placeholder(tf.float32, shape=[None, 128, 128], name='tf_x')/255
        tf_y = tf.placeholder(tf.int32, shape=[None, 101], name='tf_y')
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

        tf_x_image = tf.reshape(tf_x, shape=[-1, 128, 128, 1], name='tf_x_reshape')
        tf_y_onehot = tf_y

        h1 = self.conv_layer(tf_x_image, name='conv_1', kernel_size=(5,5), 
                            padding_mode='SAME', n_output_channels=32)
        h1_pool = tf.nn.max_pool(h1, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

        h2 = self.conv_layer(h1_pool, name='conv_2', kernel_size=(5,5), padding_mode='SAME',
                            n_output_channels=64)
        h2_pool = tf.nn.max_pool(h2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

        h3 = self.fc_layer(h2_pool, name='fc_3', n_output_units=1024,
                         activation_fn=tf.nn.relu)
        h3_drop = tf.nn.dropout(h3, keep_prob=1-self.dropout_rate, name='dropout_layer')

        h4 = self.fc_layer(h3_drop, name='fc_4', n_output_units=101, activation_fn=None)

        predictions = {
            'probabilities' : tf.nn.softmax(h4, name='probabilities'),
            'labels' : tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
        }

        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=h4, labels=tf_y_onehot
            ),
            name = 'cross_entropy_loss'
        )


        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

        correct_predictions = tf.equal(
            predictions['labels'],
            tf.cast(tf.argmax(tf_y, axis=1), tf.int32),
            name='correct_preds'
        )

        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy'
        )

    def save(self, epoch, path='model/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        print('Saving model in {}'.format(path))
        self.saver.save(
            self.sess, 
            os.path.join(path, 'cnn-model.ckpt'), 
            global_step=epoch
        )

    def load(self, path, epoch):
        print('Loading model from {}'.format(path))
        self.saver.restore(self.sess,
            os.path.join(path, 'cnn-model.ckpt-{}'.format(epoch)))


    def train(self, training_set, validation_set=None, initialize=True):
            if initialize:
                self.sess.run(self.init_op)
            
            X_data = np.array(training_set[0])
            y_data = np.array(training_set[1])

            for epoch in range(1, self.epochs+1):
                batch_gen = batch_generator(
                    X_data,
                    y_data,
                    batch_size=64,
                    shuffle=self.shuffle
                )
            
                avg_loss = 0.0
                avg_acc = 0.0
                for i, (batch_x, batch_y) in enumerate(batch_gen):
                    feed = {
                        'tf_x:0' : batch_x,
                        'tf_y:0' : batch_y,
                        'is_train:0' : True
                    }
                    loss, acc, _ = self.sess.run(
                        ['cross_entropy_loss:0', 'accuracy:0', 'train_op'],
                        feed_dict = feed
                    )
                    avg_loss += loss
                    avg_acc += acc
                self.training_loss.append(avg_loss / (i+1))
                self.training_accuracy.append(avg_acc / (i+1))
                print("="*50)
                print("Epoch {} Training Avg. Loss: {}".format(epoch, avg_loss / (i+1)))
                print("Epoch {} Training Avg. Acc: {}".format(epoch, avg_acc / (i+1)))

                if validation_set is not None:
                    X_test = validation_set[0]
                    y_test = validation_set[1]
                    batch_gen = batch_generator(
                        X_test,
                        y_test,
                        batch_size=211,
                        shuffle=self.shuffle
                    )
                    avg_loss = 0.0
                    avg_acc = 0.0
                    for i, (batch_x, batch_y) in enumerate(batch_gen):
                        feed = {
                            'tf_x:0' : batch_x,
                            'tf_y:0' : batch_y,
                            'is_train:0' : False
                        }
                        valid_loss, valid_acc = self.sess.run(
                            ['cross_entropy_loss:0', 'accuracy:0'],
                             feed_dict=feed
                        )
                        avg_loss += valid_loss
                        avg_acc += valid_acc
                    
                    self.testing_loss.append(avg_loss/6)
                    self.testing_accuracy.append(avg_acc/6)
                    print(" Validation Loss: {}".format(avg_loss/6))
                    print(" Validation Acc: {}".format(avg_acc/6))
                else:
                    print()

    def predict(self, X_test, return_proba=False):
        feed = {
            'tf_x:0' : X_test,
            'is_train:0' : False
        }

        if return_proba:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0',feed_dict=feed)

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
    
def batch_generator(X, y, batch_size=64, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size, :])

        
if __name__ == "__main__":
    X_train, name_train, y_train = label_generator("prepro_train")
    X_test, name_test, y_test = label_generator("prepro_test") # narray
    
    epochs = 20
    cnn = CNN(random_seed=123, 
            batchsize=64, 
            epochs=epochs
            )
    # cnn.load(epoch=20, path='model/')
    cnn.train(training_set=(X_train, y_train),
            validation_set=(X_test, y_test),
            initialize=True
            )
    plt.subplot(2,1,1)
    plt.plot(range(1,epochs+1), cnn.training_loss, label='training loss')
    plt.plot(range(1,epochs+1), cnn.testing_loss, label='testing loss')
    plt.xlabel('epoch')
    plt.ylabel('Cross entropy')
    plt.title('Learning Curve')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(range(1,epochs+1), cnn.training_accuracy, label='training acc')
    plt.plot(range(1,epochs+1), cnn.testing_accuracy, label='testing acc')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    # cnn.save(epoch=epochs)
    pass

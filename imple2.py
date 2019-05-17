
import os
import numpy as np
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt
from data_prepro import data_preprocess, label_generator


class CNN():
    def __init__(self,
                batchsize=64,
                epochs=10,
                learning_rate=1e-4,
                dropout_rate=0.5,
                shuffle=True,
                random_seed=None,
                regularization=False,
                reg_constant=0.01,
                batch_normalization=False
                ):
        
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
        self.regularization = regularization
        self.reg_constant = reg_constant
        self.batch_normalization = batch_normalization

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
        if self.regularization:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_constant)
            reg_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            cross_entropy_loss += reg_term

            # Obj: Regularization
            # https://www.itread01.com/content/1550615421.html
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # reg_constant = 0.5  # Choose an appropriate one.
            # cross_entropy_loss = cross_entropy_loss + reg_constant * sum(reg_losses)


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
            os.path.join(path, 'cnn-model.ckpt-{}'.format(epoch))
        )

        # from tensorflow.python.tools import inspect_checkpoint as chkp
        # chkp.print_tensors_in_checkpoint_file(
        #     os.path.join(path, 'cnn-model.ckpt-{}'.format(epoch)),
        #     tensor_name='conv_1/_weights',
        #     all_tensors=False
        # )

        # Obj: Extract weights
        # http://landcareweb.com/questions/24034/ru-he-cha-zhao-jian-cha-dian-zhong-bao-cun-de-bian-liang-ming-cheng-he-zhi
        from tensorflow.python import pywrap_tensorflow
        checkpoint_path = os.path.join(path, 'cnn-model.ckpt-{}'.format(epoch))
        self.reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        self.var_to_shape_map = self.reader.get_variable_to_shape_map()
        # for key in self.var_to_shape_map:
            # print("tensor_name: ", key)
            # print(reader.get_tensor(key))


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
                    if self.batch_normalization:
                        batch_x, batch_mean, batch_var = batch_normalization(batch_x)
                        self.batch_mean = batch_mean
                        self.batch_var = batch_var
                    
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
                print("Epoch {} Training Avg. Acc : {}".format(epoch, avg_acc / (i+1)))

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
                        if self.batch_normalization:
                            batch_x, _, _ = batch_normalization(batch_x, self.batch_mean, self.batch_var)
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
                    print("           Validation Loss : {}".format(avg_loss/6))
                    print("           Validation Acc  : {}".format(avg_acc/6))
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

            biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_channels]))

            conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
            conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
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

            biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_units]))

            layer = tf.matmul(input_tensor, weights)
            layer = tf.nn.bias_add(layer, biases, name='net_pre-activation')

            if activation_fn is None:
                return layer

            layer = activation_fn(layer)
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

def batch_normalization(X, given_mean=None, given_var=None):
    # Obj: Batch normalization
    # https://towardsdatascience.com/understanding-batch-normalization-with-examples-in-numpy-and-tensorflow-with-interactive-code-7f59bb126642
    batch_size = X.shape[0]
    img_size = X.shape[1:]
    # print(batch_size, img_size)
    mean = lambda X, batch_size: np.sum(X, axis=0) / batch_size 
    var = lambda X, batch_mean, batch_size: np.sum(np.square(X - batch_mean), axis=0) / batch_size
    if given_mean is not None:
        batch_mean = given_mean
    else:
        batch_mean = mean(X, batch_size)
    if given_var is not None:
        batch_var = given_var
    else:
        batch_var = var(X, batch_mean, batch_size)
    # print(batch_mean, batch_var)
    X = (X - batch_mean) / np.sqrt(batch_var + 1e-10)
    # new_batch_mean = mean(X, batch_size)
    # new_batch_var = var(X, new_batch_mean, batch_size)
    # print(new_batch_mean, new_batch_var)
    return X, batch_mean, batch_var
        
def main():
    aug = False
    if aug:
        X_train, name_train, y_train = label_generator("aug_train")
    else:
        X_train, name_train, y_train = label_generator("prepro_train")
    X_test, name_test, y_test = label_generator("prepro_test") # narray

    test_batch = False
    if test_batch:
        batch_gen = batch_generator(X_test, y_test, batch_size=64, shuffle=False, random_seed=42)
        batch_x, batch_y = next(batch_gen)
        # for i, (batch_x, batch_y) in enumerate(batch_gen):
        #     batch_normalization(batch_x)
        

    build_model = True
    if build_model:    
        epochs = 20
        # regularization=False, learning_rate=8e-4 -> 66%
        # regularization=True, learning_rate=2e-4 -> 64% 63%
        # regularization=True, learning_rate=2e-4, reg_constant=5e-4 -> 65% 64%
        cnn = CNN(
                batchsize=64,
                epochs=epochs,
                learning_rate=8e-4,
                dropout_rate=0,
                shuffle=True,
                random_seed=42,
                regularization=False,
                reg_constant=5e-4,
                batch_normalization=True
            )

        cnn.train(training_set=(X_train, y_train),
                validation_set=(X_test, y_test),
                initialize=True
                )
        cnn.save(epoch=epochs)

        cnn.load(epoch=epochs, path='model/')

    plot_metric = False
    if plot_metric:
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

    plot_weights = False
    if plot_weights:
        # Obj: Plot weights in probability distribution
        # https://stackoverflow.com/questions/5498008/pylab-histdata-normed-1-normalization-seems-to-work-incorrect
        # print(type(conv1), conv1.shape)
        # -> <class 'numpy.ndarray'> (5, 5, 1, 32)
        conv1 = cnn.reader.get_tensor('conv_1/_weights').flatten()
        conv2 = cnn.reader.get_tensor('conv_2/_weights').flatten()
        dense = cnn.reader.get_tensor('fc_3/_weights').flatten()
        output = cnn.reader.get_tensor('fc_4/_weights').flatten()
        bins = 100

        plt.subplot(2,2,1)
        plt.title('Histogram of conv1')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        conv1_hist_weights = np.ones_like(conv1)/float(len(conv1))
        plt.hist(conv1, bins=bins, weights=conv1_hist_weights)

        plt.subplot(2,2,2)
        plt.title('Histogram of conv2')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        conv2_hist_weights = np.ones_like(conv2)/float(len(conv2))
        plt.hist(conv2, bins=bins, weights=conv2_hist_weights)
        
        plt.subplot(2,2,3)
        plt.title('Histogram of dense1')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        dense_hist_weights = np.ones_like(dense)/float(len(dense))
        plt.hist(dense, bins=bins, weights=dense_hist_weights)
        
        plt.subplot(2,2,4)
        plt.title('Histogram of output')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        output_hist_weights = np.ones_like(output)/float(len(output))
        plt.hist(output, bins=bins, weights=output_hist_weights)

    plot_convolved = False
    if plot_convolved:
        
        from scipy import misc
        # fullpath = "train\\Motorbikes\\image_0360.jpg"
        # face = misc.imread(fullpath)
        # face = misc.imresize(face, (128, 128))
        # face = np.dot(face[...,:3], [0.2989, 0.5870, 0.1140])
        # misc.imsave("Q4.jpg", face)

        face = misc.imread("Q4.jpg")
        # plt.imshow(face, cmap=plt.cm.gray)
        filter1 = cnn.reader.get_tensor('conv_1/_weights').squeeze() # (5, 5, 1, 32) -> (5, 5, 32)
        filter2 = cnn.reader.get_tensor('conv_2/_weights') # (5, 5, 32, 64)
        filtered1 = np.zeros((128, 128, 32))
        filtered2 = np.zeros((32, 32, 64))
        for i in range(32):
            filtered1[:,:,i] = ndimage.convolve(face, filter1[:,:,i], mode='constant', cval=0.0)

        filtered1 = np.maximum(filtered1, 0) # relu
        maxpooled1 = np.zeros((32,32,32)) # max pooling
        for i in range(0, 128, 4):
            for j in range(0, 128, 4):
                for k in range(32):
                    maxpooled1[int(i/4), int(j/4), k] = np.max(filtered1[i:i+4, j:j+4, k])

        for i in range(64):
            tmp = np.zeros((32, 32, 32))
            for j in range(32):
                tmp[:,:,j] = ndimage.convolve(
                    maxpooled1[:,:,j].squeeze(),
                    filter2[:,:,j,i].squeeze(),
                    mode='constant', cval=0.0
                )
            filtered2[:,:,i] = np.sum(tmp, axis=2);
        # filtered2 = np.maximum(filtered2, 0); # relu
        
        fig1 = plt.figure(1)
        sideLen = 4
        for i in range(sideLen**2):
            plt.subplot(sideLen,sideLen,i+1)
            plt.imshow(filtered1[:,:,i], cmap=plt.cm.gray)
        fig1.show()
        
        fig2 = plt.figure(2)
        sideLen = 4
        for i in range(sideLen**2):
            plt.subplot(sideLen,sideLen,i+1)
            plt.imshow(filtered2[:,:,i], cmap=plt.cm.gray)
        fig2.show()
        input("Type anything to exit...")

    if build_model:
        del cnn

if __name__ == "__main__":
    main()

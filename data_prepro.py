import os
from os import listdir
from os.path import isfile, isdir, join
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def data_preprocess(folder=None, root=None):
    if root is not None:
        path = root

    if folder is not None:
        path += "\\" + folder
    
    files = listdir(path)
    syllable = path.split('\\')

    i = -1
    for f in files:
        i += 1
        fullpath = join(path, f)
        if isfile(fullpath):
            # print("file：", f)
            face = misc.imread(fullpath,'L')
            size = 256
            face = misc.imresize(face, (size,size))
            # print(face.shape, face.dtype)
            if len(face.shape) == 3:
                face = rgb2gray(face)

            if syllable[0] == 'test':
                # e.g. prepro_test\accordion_0.jpg
                newpath = "prepro_test" + "\\{}_{}.jpg".format(syllable[1], i)
                print(newpath)
            elif syllable[0] == 'train':
                newpath = "prepro_train" + "\\{}_{}.jpg".format(syllable[1], i)
                # print(newpath)
            misc.imsave(newpath, face)
            # plt.imshow(face,cmap=plt.cm.gray)
            # plt.show()

        elif isdir(fullpath):
            # print("folder：", f)
            # if syllable[0] == 'test':
            #     createFolder("prepro_test\\"+f)
            # elif syllable[0] == 'train':
            #     createFolder("prepro_train\\"+f)
            data_prepro(folder=f, root=path)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def label_generator(path=None):
    files = listdir(path)
    X = []
    y = []
    for f in files:
        fullpath = join(path, f)
        if isfile(fullpath):
            label = '_'.join((f.split('_'))[:-1])
            face = misc.imread(fullpath,'L')
            X.append(face)
            y.append(label)
            # print("file：", label)
            # plt.imshow(face,cmap=plt.cm.gray)
            # plt.show()

    values = array(y)
    
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    
    # invert first example
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)
    
    return np.asarray(X), np.asarray(y), onehot_encoded


if __name__ == "__main__":
    ## prepose training set and testing set
    # data_preprocess(root="test")
    # data_preprocess(root="train")
    X, y, onehot_encoded = label_generator("prepro_test")
    # X, y, classes = label_generator("prepro_train")
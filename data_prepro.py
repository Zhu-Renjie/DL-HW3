import os
from os import listdir
from os.path import isfile, isdir, join
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np


def data_prepro(folder=None, root=None):
    if root is not None:
        path = root

    if folder is not None:
        path += "\\" + folder
    
    files = listdir(path)
    syllable = path.split('\\')

    for f in files:
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
                newpath  = "prepro_test" + fullpath[4:]
            elif syllable[0] == 'train':
                newpath  = "prepro_train" + fullpath[5:]
            misc.imsave(newpath, face)
            # plt.imshow(face,cmap=plt.cm.gray)
            # plt.show()

        elif isdir(fullpath):
            # print("folder：", f)
            if syllable[0] == 'test':
                createFolder("prepro_test\\"+f)
            elif syllable[0] == 'train':
                createFolder("prepro_train\\"+f)
            data_prepro(folder=f, root=path)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if __name__ == "__main__":
    ## prepose training set and testing set
    # data_prepro(root="test")
    # data_prepro(root="train")
    pass
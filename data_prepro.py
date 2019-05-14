import os
from os import listdir
from os.path import isfile, isdir, join
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np


def data_prepro(folder=None):
    path = "prepro_test"
    # path = "train"

    if folder is not None:
        path += "\\" + folder
    files = listdir(path)

    for f in files:
        fullpath = join(path, f)
        if isfile(fullpath):
            # print("file：", f)
            face = misc.imread(fullpath,'L')
            # face = misc.imresize(face, (256,256))
            # print(face.shape, face.dtype)
            if len(face.shape) == 3:
                face = rgb2gray(face)

            # newpath  = "prepro_test" + fullpath[4:]
            # newpath  = "prepro_train" + fullpath[5:]
            # misc.imsave(newpath, face)
            # plt.imshow(face,cmap=plt.cm.gray)
            # plt.show()

        elif isdir(fullpath):
            # print("folder：", f)
            data_prepro(f)
            # createFolder("prepro_test\\"+f)
            # createFolder("prepro_train\\"+f)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if __name__ == "__main__":
    data_prepro()
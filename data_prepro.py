import os
from os import listdir
from os.path import isfile, isdir, join
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def data_preprocess(folder=None, root=None, aug=False):
    if root is not None:
        path = root

    if folder is not None:
        path += "\\" + folder
    
    files = listdir(path)
    syllable = path.split('\\')

    i = -1
    for f in files:
        fullpath = join(path, f)
        if isfile(fullpath):
            # print("file：", f)
            face = misc.imread(fullpath,'L')
            size = 128
            face = misc.imresize(face, (size,size))
            # print(face.shape, face.dtype)
            if len(face.shape) == 3:
                face = rgb2gray(face)

            if syllable[0] == 'test':
                # e.g. prepro_test\accordion_0.jpg
                if aug:
                    pass
                else:
                    newpath = "prepro_test" + "\\{}_{}.jpg".format(syllable[1], i)
                    i += 1
                    misc.imsave(newpath, face)

            elif syllable[0] == 'train':
                if aug:
                    newpath = "aug_train" + "\\{}_{}.jpg".format(syllable[1], i)
                    i += 1
                    misc.imsave(newpath, face)

                    newpath = "aug_train" + "\\{}_{}.jpg".format(syllable[1], i)
                    
                    noise = [('Gaussian', 10), ('Salt&Pepper', 0.01)][np.random.randint(2)]
                    angle = 180 * np.random.random() - 90 # -90 ~ +90
                    face2 = data_augmentation(  img=face,
                                                flip=np.random.randint(2),
                                                rotate=angle,
                                                noise=noise
                                            )
                    i += 1
                    misc.imsave(newpath, face2)
                else:
                    newpath = "prepro_train" + "\\{}_{}.jpg".format(syllable[1], i)
                    i += 1
                    misc.imsave(newpath, face)
            
            # plt.imshow(face,cmap=plt.cm.gray)
            # plt.show()

        elif isdir(fullpath):
            # print("folder：", f)
            # if syllable[0] == 'test':
            #     createFolder("prepro_test\\"+f)
            # elif syllable[0] == 'train':
            #     createFolder("prepro_train\\"+f)
<<<<<<< HEAD
            data_preprocess(folder=f, root=path)
=======
            data_preprocess(folder=f, root=path, aug=aug)
>>>>>>> origin/master

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
    
    # Obj: One-hot encoding
    # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    # binary encode
    values = values.reshape(len(values), 1)
    onehot_encoded = OneHotEncoder(categories='auto',sparse=False).fit_transform(values)
    # print(onehot_encoded)
    
    # invert the one-hot to label
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)
    
    return np.asarray(X), np.asarray(y), onehot_encoded

def data_augmentation(  path=None,
                        img=None, 
                        flip=False,
                        rotate=None,
                        noise=None
                        ):
    if img is None:
        face = misc.imread(path,'L')
    else:
        face = img
    
    if flip:
        face = np.fliplr(face)
    if rotate is not None:
        face = ndimage.rotate(face, rotate, reshape=False)
    if noise is not None:
        if noise[0] == 'Gaussian':
            face = face + np.random.normal(0, noise[1], face.shape)
            face = np.clip(face, 0, 255)
        if noise[0] == 'Salt&Pepper':
            # Obg: Salt and Pepper noise
            # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
            H, W = face.shape
            prob = noise[1]
            thres = 1 - prob
            for i in range(H):
                for j in range(W):
                    rdn = np.random.random()
                    if rdn < prob:
                        face[i][j] = 0
                    elif rdn > thres:
                        face[i][j] = 255
    # plt.imshow(face,cmap=plt.cm.gray)
    return face


if __name__ == "__main__":
    ## prepose training set and testing set
    # data_preprocess(root="test")
    # data_preprocess(root="train")
    # X, y, onehot_encoded = label_generator("prepro_test")
    # X, y, classes = label_generator("prepro_train")
<<<<<<< HEAD
=======
    # data_augmentation(  path="Q4.jpg",
    #                     rotate=90,
    #                     # noise=('Gaussian', 10),
    #                     noise=('Salt&Pepper', 0.01)
    #                 )
    data_preprocess(root="train", aug=True)
>>>>>>> origin/master
    pass
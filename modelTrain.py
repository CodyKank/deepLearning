#!/usr/bin/python
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.models import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, Convolution3D, UpSampling2D
from keras.utils import np_utils
import skimage
from skimage import io, color
from keras.optimizers import SGD
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img

"""
Colorizing a Gray Scale image utilizing LAB format.
L band - Brightness, is the 0 band which will be passed into the network for an input. Range of values: [0,100]
A and B band - Deal with colors. Range of values: [-128, 128] (give or take)

Note on training: The A and B bands will have to all be divided by 128 to make their values fall between [-1, and 1]. This is so
the output of the network can be mapped to the correct values, as it outputs [-1,1] so the output values can be multiplied by
128 to create the desired values (in theory).
"""


def main():


    trainGen = get_images('/data/mit1/good_train/*.jpg') # Creating generator for training images
    validationGen = get_images('/data/mit1/valid_good/*jpg') # Creating generator for validation images

    model = Sequential()
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same', strides=2,input_shape=(225, 225,1)))
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2))) # Adding a pooling of features
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D(size=(2,2))) # Upsampling to increase size after pooling
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(Convolution2D(16, (3,3), activation='relu'))
    model.add(Convolution2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(4,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(1250,(9,9), activation='tanh'))
    sgd = SGD(lr=0.04, momentum=0.6)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    print( model.output_shape)
    
    model.fit_generator(trainGen, validation_data = validationGen, steps_per_epoch = 3115, validation_steps= 401, epochs=4)
    model.save('deepCorloziationWeights.h5')
#^-----------------------------------------------------------------------------main()
    

def splitArray(arrayToSplit):
    """Function to split an array of [225,225] into 9 equal arrays of [75,75].
    An image is broken up into a grid of 75x75 portions.
    IN: A [225,225] list or np.array
    OUT: A python list of 9 np.arrays of size [75,75]"""

    size = 9
    desired = [] # trying just a python list

    # I will go 0 to 1250 due to 25 blocks of 9x9 per row equaling 625 blocks for each band of A
    # and B, so 1250 total blocks to grab.

    #Starting with once per each band, only 625 blocks
    for j in range(0,25):
        for i in range (0, 25):
            desired.append(arrayToSplit[(size*i):((i+1)*size), size*j:((j+1)*size)])

    return desired
#^-----------------------------------------------------------------------------splitArray(arrayToSplit)


def mergeArray(arrayToMerge):
    """Takes in an array which is meant to be 3 dimensions and containing the different grid-portions
    of A and B bands. It mashes each section next to each other in the proper order from the given
    array and returns a 225x225 array which is a complete A or B band."""

    size = 9
    placematCount = 0
    desired = np.zeros((225,225))
    for j in range(0,25):
        for i in range(0,25):
            desired[(size*i):((i+1)*size), size*j:((j+1)*size)] = arrayToMerge[:,:,placematCount]
            placematCount += 1
    return desired
#^-----------------------------------------------------------------------------mergeArray(arrayToMerge)    

def get_images(path):
    """ Generator to obtain and yield an image from a directory. YIELDS a tuple of an input band and desired bands (A + B) in a
    flattened array
    IN: A string denoting the full path to the images, with a wildcard ie /data/mit1/images256/meow/*.jpg"""
    fileList = glob.glob(path)
    listSize = len(fileList)
    fileCount = 0 # which file we're working with
    while True:
        filename = fileList[fileCount]
        try:
            image = skimage.color.rgb2lab(skimage.io.imread(filename))
        except ValueError:
            continue
        Lband = np.zeros((1, 225,225,1))
        Lband[0, :,:,0] = image[0:225,0:225,0]/100.0
        A_List =  splitArray(image[0:225, 0:225, 1]/128.0)
        B_List = splitArray(image[0:225,0:225,2]/128.0)
        imageLabel = np.zeros((1,9,9,1250))

        for i in range(0,1250):
            if i < 625:
                imageLabel[0,:,:,i] = np.array(A_List[i], dtype=float)
            else:
                imageLabel[0,:,:,i] = np.array(B_List[i-625],dtype=float)

        fileCount += 1 # Moving to the next file
        if fileCount == listSize:
            fileCount = 0 # Resetting fileCount
        yield(Lband, imageLabel)
#^-----------------------------------------------------------------------------get_images(path)

#Standard boiler plate to run as main
if __name__ == '__main__':
    main()

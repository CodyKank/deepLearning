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

Operating the network: This network is thus far only created to output a correct shape. There are no weights associated
yet. The plan is to train the network to properly adjust the weights to make the output shape the A and B bands.

Training on A and B bands: Getting the right output shape is a pain. The current shape is (75,75, 6). This means the A band
will be placed witin (75, 75, 0) through (75,75,2) and B will be placed in the res (3-5). This way, both bands can be 
discovered and trained for at once.

Note on training: The A and B bands will have to all be divided by 128 to make their values fall between [-1, and 1]. This is so
the output of the network can be mapped to the correct values, as it outputs [-1,1] so the output values can be multiplied by
128 to create the desired values (in theory).
"""


def main():


    #trainGen = get_Arrayimages('/data/mit1/images256/p/newPastures/*.jpg') # Creating generator for training images
    #validationGen = get_Arrayimages('/data/mit1/images256/p/validation/*.jpg') # Creating generator for validation images
    trainGen = get_images('/data/mit1/images256/train/*.jpg') # Creating generator for training images
    validationGen = get_images('/data/mit1/images256/validation/*.jpg') # Creating generator for validation images

    """
    # Loading an initial test image
    image = skimage.io.imread('.jpg')

    # Moving the original image into LAB format
    labImage = skimage.color.rgb2lab((image))
    #labImage.reshape(225,225,3)

    # Grabbing just the L band (brightness)
    L_band = labImage[:,:,0]

    # Moving the L_band list into an np array
    L_input = np.zeros((1, 225,225,1))
    L_input[0,:,:,0] = L_band[0:225,0:225]

    #L_input = np.array(L_band, dtype=float)

    #image = np.array(image, dtype=float)
    Desired_Band_A = np.array(labImage[0:225,0:225,1], dtype=float)/128.0
    Desired_Band_B = np.array(labImage[0:225,0:225,2], dtype=float)/128.0

    # For the labels, must split 225x225 into 1250 9x9 blocks !!!
    imageLabel = np.zeros((1,9,9,1250))

    #Splitting bands A and B into 3 equal sized arrays len=75
    A_List = []


    # Splitting each band into 625 9x9 blocks
    A_List = splitArray(Desired_Band_A)
    B_List = splitArray(Desired_Band_B)
    
    # Going through each list and creating a 75,75,6 for the two bands
    for i in range(0,1250):
        if i < 625:
            imageLabel[0,:,:,i] = np.array(A_List[i], dtype=float)
        else:
            imageLabel[0,:,:,i] = np.array(B_List[i-625],dtype=float)


    """
    model = Sequential()
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same', strides=2,input_shape=(225, 225,1)))
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2))) # Adding a pooling of features
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D(size=(2,2))) # Upsampling to increase size after pooling
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(3,3)))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    #model.add(UpSampling2D(size=(2,2)))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(Convolution2D(16, (3,3), activation='relu'))
    model.add(Convolution2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(4,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(1250,(10,10), activation='tanh'))
    #model.compile(optimizer='rmsprop', loss='mse')
    sgd = SGD(lr=0.01, momentum=0.5)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    print( model.output_shape)
    
    #model.fit_generator(trainGen, validation_data = validationGen, steps_per_epoch = 10, validation_steps=5, epochs=10)
    model.fit_generator(trainGen, steps_per_epoch = 10, epochs=10)
    model.save('deepCorloziationWeights.h5')
    #model.fit(x=L_input, y=imageLabel, batch_size=1, epochs=200)
    """
    
    # Testing the image we just trained on
    testing = model.predict(L_input)
    testing[0,:,:,:] *= 128.0
    resA = mergeArray(testing[0,:,:,0:625])
    resB = mergeArray(testing[0,:,:,625:])

    resA[0:75, 0:75] = (testing[0,:,:,0])
    resA[0:75, 75:150] = (testing[0,:,:,1])
    resA[0:75, 150:225] = testing[0,:,:,2]
    resA[75:150, 0:75] = testing[0,:,:,3]
    resA[75:150, 150:225] = testing[0,:,:,4]
    resA[75:150, 150:225] = testing[0,:,:,5]
    resA[150:225, 0:75] = testing[0,:,:,6]
    resA[150:225, 75:150] = testing[0,:,:,7]
    resA[150:225, 150:225] = testing[0,:,:,8]

    resB[0:75, 0:75] = testing[0,:,:,9]
    resB[0:75, 75:150] = testing[0,:,:,10]
    resB[0:75, 150:225] = testing[0,:,:,11]
    resB[75:150, 0:75] = testing[0,:,:,12]
    resB[75:150, 150:225] = testing[0,:,:,13]
    resB[75:150, 150:225] = testing[0,:,:,14]
    resB[150:225, 0:75] = testing[0,:,:,15]
    resB[150:225, 75:150] = testing[0,:,:,16]
    resB[150:225, 150:225] = testing[0,:,:,17]
    

    trialImage = np.zeros((225,225,3))
    trialImage[:,:,0] = L_band[0:225, 0:225]
    trialImage[:,:,1] = resA
    trialImage[:,:,2] = resB
    
    saveImage = skimage.color.lab2rgb(trialImage)
    skimage.io.imsave('meow.jpg', saveImage)
    """

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
    
    
    """retList = []
    retList.append(arrayToSplit[0:75, 0:75])
    retList.append(arrayToSplit[0:75, 75:150])
    retList.append(arrayToSplit[0:75, 150:225])
    retList.append(arrayToSplit[75:150, 0:75])
    retList.append(arrayToSplit[75:150, 75:150])
    retList.append(arrayToSplit[75:150, 150:225])
    retList.append(arrayToSplit[150:225, 0:75])
    retList.append(arrayToSplit[150:225, 75:150])
    retList.append(arrayToSplit[150:225, 150:225])
    return retList"""
#^-----------------------------------------------------------------------------splitArray(arrayToSplit)


def mergeArray(arrayToMerge):
    """Comments """

    size = 9
    placematCount = 0
    desired = np.zeros((225,225))
    for j in range(0,25):
        for i in range(0,25):
            desired[(size*i):((i+1)*size), size*j:((j+1)*size)] = arrayToMerge[:,:,placematCount]
            placematCount += 1
    return desired
    
def get_Arrayimages(path):
    """Returning numpy arrays of both the images and corresponding labels or desired data."""

    attempt = np.zeros((1,225,225,1,10)) # Creating initialization for np array of input images
    attemptOut = np.zeros((1, 9,9,1250,10)) # Creating initialization for np array of labels
    count = 0
    for filename in glob.glob(path):
        try:
            image = skimage.color.rgb2lab(skimage.io.imread(filename))
        except ValueError:
            continue # Skipping images which can't be loaded
        Lband = np.zeros((1, 225,225,1))
        Lband[0, :,:,0] = image[0:225,0:225,0]
        A_List =  splitArray(image[0:225, 0:225, 1])
        B_List = splitArray(image[0:225,0:225,2])
                
        for i in range(0,1250):
            if i < 625:
                attemptOut[0,:,:,i,count] = np.array(A_List[i], dtype=float)
            else:
                attemptOut[0,:,:,i,count] = np.array(B_List[i-625],dtype=float)


        count += 1


def get_images(path):
    """ Generator to obtain and yield an image from a directory. YIELDS a tuple of an input band and desired bands (A + B) in a
    flattened array
    IN: A string denoting the full path to the images, with a wildcard ie /data/mit1/images256/meow/*.jpg"""
    fileList = glob.glob(path)
    listSize = len(fileList)
    fileCount = 0 # which file we're working with
    #for filename in glob.glob(path):
    while True:
        filename = fileList[fileCount]
        try:
            image = skimage.color.rgb2lab(skimage.io.imread(filename))
        except ValueError:
            continue
        Lband = np.zeros((1, 225,225,1))
        Lband[0, :,:,0] = image[0:225,0:225,0]
        #trainTarget = np.zeros((1, 100352))
        #trainTarget[0, 0:50176] = np.array(image[0:224,0:224,1]).flatten() # A band ***May need to flatten() these?
        #trainTarget[0, 50176:] = np.array(image[0:224,0:224,2]).flatten() # B band
        A_List =  splitArray(image[0:225, 0:225, 1])
        B_List = splitArray(image[0:225,0:225,2])
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


#Standard broiler plate to run as main
if __name__ == '__main__':
    main()

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import skimage
from skimage import io, color 
from keras.optimizers import SGD
import glob


# Loading all training images into a numpy array
#Glob the directory, and load all images and do the following in a loop!
# (Num images, size, size, numBands)
#trainInput = np.zeros((60396, 224, 224, 3))
#trainTarget = np.zeros((60396, 100352))
#listCount = 0
# Grabbing all jpg images within train directory
"""This is set up in a way where each image will be denoted by the first dimension as desired by keras.
Thereafter, the output data (labels) will be the flattened arrays of each desired output from the network."""
"""for filename in glob.glob('/data/mit1/images256/train/*.jpg'):
    image = skimage.color.rgb2lab(skimage.io.imread(filename))
    trainInput[listCount, :, :, 0] = image[0:224,0:224,0] 
    trainTarget[listCount, 0:50176] = image[0:224,0:224,1] # A band ***May need to flatten() these?
    trainTarget[listCount, 50176:] = image[0:224,0:224,2] # B band"""

def get_images(path):
    """ Generator to obtain and yield an image from a directory. YIELDS a tuple of an input band and desired bands (A + B) in a
    flattened array
    IN: A string denoting the full path to the images, with a wildcard ie /data/mit1/images256/meow/*.jpg"""
    for filename in glob.glob(path):
        image = skimage.color.rgb2lab(skimage.io.imread(filename))
        Lband = np.zeros((1, 224,224,3))
        Lband[0, :,:,0] = image[0:224,0:224,0]
        trainTarget = np.zeros((1, 100352))
        trainTarget[0, 0:50176] = np.array(image[0:224,0:224,1]).flatten() # A band ***May need to flatten() these?
        trainTarget[0, 50176:] = np.array(image[0:224,0:224,2]).flatten() # B band
        yield(Lband, trainTarget)

"""
listCount = 0
validationIn = np.zeros((27894, 224, 224, 3))
validationTarg = np.zeros((27894, 100352))
validationData = (validationIn, validationData) # The tuple of these two
# Need to repeat above for validation set
for filename in glob.glob('/data/mit1/images256/validation/*.jpg'):
    image = skimage.color.rgb2lab(skimage.io.imread(filename))
    validationIn[listCount, :, :, 0] = image[0:224,0:224,0] 
    validationTarg[listCount, 0:50176] = image[0:224,0:224,1] # A band ***May need to flatten() these?
    validationTarg[listCount, 50176:] = image[0:224,0:224,2] # B band
"""
trainGen = get_images('/data/mit1/images256/train/*.jpg') # Creating generator for training images
validationGen = get_images('/data/mit1/images256/validation/*.jpg') # Creating generator for validation images


# Change model input to accept the new arrays of trainInput and trainTarget, set validation data as well.

"""
# Loading an initial test image
image = skimage.io.imread('fireTruck.jpg')

# Moving the original image into LAB format
labImage = skimage.color.rgb2lab((image))
#labImage.reshape(225,225,3)

# Grabbing just the L band (brightness)
L_band = labImage[:,:,0]
# Moving the L_band list into an np array
L_input = np.zeros((1, 224,224,3))
L_input[0,:,:,0] = L_band[0:224,0:224]

#desireBands = np.zeros(131072)
desireBands = np.zeros((1,100352)) # creating the desired output matrix

#aBand = np.array(labImage[:,:,1]).flatten()
#bBand = np.array(labImage[:,:,2]).flatten()
aBand = np.array(labImage[:,:,1])
bBand = np.array(labImage[:,:,2])

#desireBands[0:65536] = aBand
#desireBands[65536:] = bBand
desireBands[0,0:50176] = (aBand[0:224, 0:224]).flatten()/128.0
desireBands[0,50176:] = (bBand[0:224,0:224]).flatten()/128.0
"""
sgd = SGD(lr=0.01, momentum=0.5)


vgg16 = VGG16(weights='imagenet', input_shape=(224,224,3))

model = Sequential()
model.add(vgg16)

for layer in model.layers:
    if layer.name in ['fc1', 'fc2', 'logit']:
        continue
    layer.trainable = False

#model.add(Dense(131072, activation = 'tanh'))
model.add(Dense(100352, activation = 'tanh'))

model.summary()

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

#model.fit(x=trainInput, y=trainTarget, validation_data=validationData, batch_size=32, epochs=100)
model.fit_generator(trainGen, validation_data = validationGen, steps_per_epoch = 60396, validation_steps= 27894, epochs=100)

testing = model.predict(x = L_input)

testing = testing.transpose()
testingA = testing[0:50176,:].reshape((224,224))*127.0*127.0
testingB = testing[50176:,:].reshape((224,224))*127.0*127.0

testingFull = np.zeros((224,224,3))
testingFull[:,:,0] = L_band[0:224,0:224]


testingFull[:,:,1] = testingA
testingFull[:,:,2] = testingB

rgbsave = skimage.color.lab2rgb(testingFull)

skimage.io.imsave('attempt1.jpg', skimage.color.lab2rgb(testingFull))


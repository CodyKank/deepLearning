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
#GLob the directory, and load all images and do the following in a loop!


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

sgd = SGD(lr=0.01, momentum=0.5)


vgg16 = VGG16(weights='imagenet')

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

model.fit(x=L_input, y=desireBands, batch_size=1, epochs=100)

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


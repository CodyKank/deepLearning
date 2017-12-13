#!/usr/bin/python

from keras.models import load_model
import skimage, sys
from skimage import io, color
import numpy as np

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
#------------------------------------------------------------------------------mergeArray(arrayToMerge)
    
"""Testing the weights as trained by file modelTrain.py. The weights are loaded and then testing against an
   image which is entered on the cmd line after calling this file itself.
   
   EXAMPLE: ./testModel.py islet.jpg && display trainTest.jpg """
model = load_model('deepCorloziationWeights.h5')

# Loading an initial test image
image = skimage.io.imread(str(sys.argv[1])) # Grabbing an image off of the cmd line
# Moving the original image into LAB format
labImage = skimage.color.rgb2lab((image))

# Grabbing just the L band (brightness)
L_band = labImage[:,:,0]

# Moving the L_band list into an np array
L_input = np.zeros((1, 225,225,1))
L_input[0,:,:,0] = L_band[0:225,0:225]

#L_input = np.array(L_band, dtype=float)

# Testing the image we just trained on
testing = model.predict(L_input)
resA = mergeArray(testing[0,:,:,0:625]*128.0) # multiplying each value by 128 to get back to [-128,128]
resB = mergeArray(testing[0,:,:,625:]*128.0)  # values were [-1,1].

trialImage = np.zeros((225,225,3))
trialImage[:,:,0] = L_band[0:225, 0:225]
trialImage[:,:,1] = resA
trialImage[:,:,2] = resB
    
saveImage = skimage.color.lab2rgb(trialImage) # Back to RGB so we can save as JPG
skimage.io.imsave('trainTest.jpg', saveImage)

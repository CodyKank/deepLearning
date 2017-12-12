#!/usr/bin/python
from keras.models import load_model
import skimage, sys
from skimage import io, color
import numpy as np

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

model = load_model('deepCorloziationWeights.h5')

# Loading an initial test image
#image = skimage.io.imread('islet.jpg')
image = skimage.io.imread(str(sys.argv[1]))
# Moving the original image into LAB format
labImage = skimage.color.rgb2lab((image))
#labImage.reshape(225,225,3)

# Grabbing just the L band (brightness)
L_band = labImage[:,:,0]

# Moving the L_band list into an np array
L_input = np.zeros((1, 225,225,1))
L_input[0,:,:,0] = L_band[0:225,0:225]

#L_input = np.array(L_band, dtype=float)

# Testing the image we just trained on
testing = model.predict(L_input)
#testing[0,:,:,:] *= 128.0
resA = mergeArray(testing[0,:,:,0:625]*128.0)
resB = mergeArray(testing[0,:,:,625:]*128.0)

"""resA[0:75, 0:75] = (testing[0,:,:,0])
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
"""    

trialImage = np.zeros((225,225,3))
trialImage[:,:,0] = L_band[0:225, 0:225]
trialImage[:,:,1] = resA
trialImage[:,:,2] = resB
    
saveImage = skimage.color.lab2rgb(trialImage)
skimage.io.imsave('trainTest.jpg', saveImage)

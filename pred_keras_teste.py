
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:14:44 2018

@author: v3n0w
"""

import tensorflow as tf
import imageio
import numpy as np
from sklearn.metrics import recall_score
import glob 
import matplotlib.pyplot as plt

ImageTest = glob.glob('Teste1/Images/*.png')
ImagesLabel = glob.glob('Teste1/Labels/*.png')

y_all_train = np.zeros((len(ImageTest), 256, 256, 1))

for i, imageFile in enumerate(ImageTest):
    label = imageio.imread(ImagesLabel[i])
    label = np.reshape(label[:,:,0], (256, 256, 1))
    label = label > 127
    y_all_train[i, :,:,:] = label

model = tf.keras.models.load_model('model.h5')

TP = 0
TN = 0
FP = 0
FN = 0

val_y = y_all_train

save_dir = 'ResultTeste/'

for i, imageFile in enumerate(ImageTest):
    
    image = imageio.imread(imageFile)
    folder_0,folder_1,imageName = imageFile.split("/")
    image = np.reshape(image, (1, 256, 256, 3))
    image = image / 255.
    #print(image) 
    #result = model.predict([image, np.ones((1, 256, 256, 1))])
    result =  model.predict([image])
    result = np.round(result)
    imageio.imsave(result_image, result[0, :, :, 0])

    result[0, :, :,  0] = result[0, :, :,  0] 
    result_image = save_dir + imageName
    image[0, :, :, 1] = result[0, :, :,  0]
    plt.figure()
    plt.imshow(result[0, :, :, 0])
    plt.show()
    
    '''
    plt.figure()
    plt.imshow(image[0, :, :, :])
    plt.figure()

    image[0, :, :, 1] = result[0, :, :, 0] < 0.5
    plt.imshow(image[0, :, :, :])


plt.figure()
plt.imshow(image[0, :, :, :])
plt.figure()
plt.imshow(result[0, :, :, 0])
plt.figure()
image[0, :, :, 1] = result[0, :, :, 0] > 127
plt.imshow(image[0, :, :, :])
'''

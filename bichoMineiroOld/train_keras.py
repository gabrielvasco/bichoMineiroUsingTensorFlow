# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:13:54 2018

@author: v3n0w
"""
import tensorflow as tf
import glob 
import imageio
import numpy as np


ImagesTrain = glob.glob('TrainSet/Train/Images/*.png')
LabelsTrain = glob.glob('TrainSet/Train//Labels/*.png')

ImagesVal = glob.glob('TrainSet/Test/Images/*.png')
LabelsVal = glob.glob('TrainSet/Test/Labels/*.png')

print(ImagesTrain[37] + ' == '+ LabelsTrain[37])
print('1 This ^ should be the same.')

print(ImagesVal[3] + ' == '+ LabelsVal[3])
print('2 This ^ should be the same.')

x_all_train = np.zeros((len(ImagesTrain), 256, 256, 3))
y_all_train = np.zeros((len(ImagesTrain), 256, 256, 1))

x_all_val = np.zeros((len(ImagesVal), 256, 256, 3))
y_all_val = np.zeros((len(ImagesVal), 256, 256, 1))


for i, imageFile in enumerate(ImagesTrain):
       
    image = imageio.imread(imageFile)

    image = np.reshape(image, (256, 256, 3)) 
    image = image / 255.
    
    label = imageio.imread(LabelsTrain[i])
    
    label = np.reshape(label[:,:,0], (256, 256, 1))
    
    label = label > 127
    
    x_all_train[i, :,:,:] = image
    y_all_train[i, :,:,:] = label

for i, imageFile in enumerate(ImagesVal):
    
    image = imageio.imread(imageFile)

    image = np.reshape(image, (256, 256, 3)) 
    image = image / 255.
    
    label = imageio.imread(LabelsVal[i])
    
    label = np.reshape(label[:,:,0], (256, 256, 1))
    
    label = label > 127
    
    x_all_val[i, :,:,:] = image
    y_all_val[i, :,:,:] = label
    

model = tf.keras.models.Sequential()

regu = tf.keras.regularizers.l2(0.000000)

inp = tf.keras.layers.Input((256, 256, 3))
net = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3), padding = 'same', kernel_regularizer=regu)(inp)
down_1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(down_1)

net = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
down_2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(down_2)

net = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
down_3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(down_3)

net = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
down_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(down_4)

net = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding = 'same')(net)
net = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding = 'same')(net)

net = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',  activation='relu', bias_initializer='zeros', kernel_regularizer=regu)(net)
net = tf.keras.layers.Concatenate()([net, down_4])
net = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)

net = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',  activation='relu', bias_initializer='zeros', kernel_regularizer=regu)(net)
net = tf.keras.layers.Concatenate()([net, down_3])
net = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)

net = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',  activation='relu', bias_initializer='zeros', kernel_regularizer=regu)(net)
net = tf.keras.layers.Concatenate()([net, down_2])
net = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)

net = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same',  activation='relu', bias_initializer='zeros', kernel_regularizer=regu)(net)
net = tf.keras.layers.Concatenate()([net, down_1])
net = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)
net = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding = 'same', kernel_regularizer=regu)(net)

net = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding = 'same', bias_initializer='zeros')(net)

inp_mask = tf.keras.layers.Input((256, 256, 1))
net = tf.keras.layers.Multiply()([net, inp_mask])

model = tf.keras.models.Model(inputs=[inp, inp_mask], outputs = net)

model.compile(loss='binary_crossentropy', optimizer='adam')


train_x = x_all_train
val_x = x_all_val
train_y = y_all_train
val_y = y_all_val
epochs = 30
n = 30
x = 0

test_mask_train = np.ones((train_x.shape[0], 256, 256, 1))
test_mask_val = np.ones((val_x.shape[0], 256, 256, 1))

max_acc_val = - np.inf

for i in range(epochs):
    
    print('Starting Epoch %d' %(i))
    
    loss_pblock = (np.random.rand(train_x.shape[0], 16, 16, 1) > 0.).astype(np.float)
    loss_weights = np.zeros((train_x.shape[0], 256, 256, 1))
    from skimage.transform import resize

    for j in range(train_x.shape[0]):
        loss_weights[j,...] = resize(loss_pblock[j, ...], (256, 256), mode='reflect', order = 0)
        
    train_y_m = train_y * loss_weights

    model.fit([train_x, loss_weights], train_y_m, 10, 1, shuffle=True)
    
    train_pred = model.predict([train_x, test_mask_train])
    
    train_acc = np.mean(np.isclose(np.round(train_pred), train_y))
    
    val_pred = model.predict([val_x, test_mask_val])
    
    val_acc = np.mean(np.isclose(np.round(val_pred), val_y))
    x = x + 1
   
    if max_acc_val < val_acc:
        print('Best val found!!')
        model.save('model.h5')
        max_acc_val = val_acc
        x = 0
        
    print('Train acc: %f, Val acc %f' %(train_acc, val_acc))
    
    if (x == n) & (max_acc_val > val_acc):
        print('Model did not improve for %d epochs' %(n))
        print('Stopping training')

        exit()

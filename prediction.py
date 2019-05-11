import tensorflow as tf
import imageio
import numpy as np
from sklearn.metrics import recall_score
import glob

# Comando para imprimir os objetos completos
#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(suppress=True)

ImageTest = glob.glob('Teste/Images/*.png')
ImagesLabel = glob.glob('Teste/Labels/*.png')

print(ImageTest[0] + ' == '+ ImagesLabel[0])

y_all_train = np.zeros((len(ImageTest), 256, 256, 1))

for i, imageFile in enumerate(ImageTest):
    label = imageio.imread(ImagesLabel[i])
    label = np.reshape(label[:,:,0], (256, 256, 1))
    label = label > 127
    #print("Img label:" + str( ImagesLabel[i] ) + " / label: " + str( label[150, 100][0]) )
    y_all_train[i, :,:,:] = label

model = tf.keras.models.load_model('model.h5')

TP = 0 # Sick plant correctly identified as sick.
TN = 0 # Healthy plant correctly identified as healthy
FP = 0 # Healthy plant incorrectly identified as sick.
FN = 0 # Sick plant incorrectly identified as healthy.

val_y = y_all_train

save_dir = 'ResultTeste/'

for i, imageFile in enumerate(ImageTest):
    
    image = imageio.imread(imageFile)
    image_tst = imageio.imread(imageFile)

    folder_0,folder_1,imageName = imageFile.split("/")
    image = np.reshape(image, (1, 256, 256, 3))
    image = image / 255.
   
    #result = model.predict([image, np.ones((1, 256, 256, 1))])
    #result = model.predict([image, np.ones((1, 256, 256, 1))])
    result =  model.predict([image])
    #print(result)
    #print(result.shape)
    result = np.round(result)
    #print(result)
    #print(np.any(result == 0))

    #print("result: " + str(result.shape))
    #print("val_y: " + str(val_y.shape))

    #print("Img :" + str( ImagesLabel[i] ) + " / label: " + str( val_y[i, 150, 100][0]) + " / result: " + str( result[0, 150, 100][0] ))
	
    current_TP = 0
    current_TN = 0
    current_FP = 0
    current_FN = 0

    for j in range(result.shape[1]):
        for k in range(result.shape[2]):
            #print(str(result[0, j, k][0]) + " == "+ str(val_y[i, j, k][0]))
            if result[0, j, k][0] == 0 and val_y[i, j, k][0] == 0:
                current_TP += 1
                image_tst[j, k , 2] = 255
            elif result[0, j, k][0] == 1. and val_y[i, j, k][0] == 1:
                current_TN += 1
            elif result[0, j, k][0] == 1 and val_y[i, j, k][0] == 0:
                current_FP += 1
                image_tst[j, k , 1] = 255
            elif result[0, j, k][0] == 0 and val_y[i, j, k][0] == 1:
                current_FN += 1
                image_tst[j, k , 0] = 255
            #print(result[0, j, k][0])

    #print(current_TP)
    TP = TP + current_TP
    TN = TN + current_TN
    #print(current_FP)
    FP = FP + current_FP
    #print(current_FN)
    FN = FN + current_FN
    
    total = current_TP + current_TN + current_FP + current_FN
    
    print('Total number: %d'  %(total))
    print('Acc: %f'  %((current_TP + current_TN) / total))
    
    result_image = save_dir + imageName
    evaluation = save_dir + 'evaluation' + imageName
    
    imageio.imsave(result_image, result[0, :, :, 0])
    imageio.imsave(evaluation, image_tst)

print("TP: " + str(TP))
print("FP: " + str(FP))
print("TN: " + str(TN))
print("FN: " + str(FN))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall / (precision + recall))
acc = (TP + TN) / (TP + TN + FP + FN)

print('Precision >')
print(precision)
print('Recall >')
print(recall)
print('F-Measure >')
print(f1)
print('Test Accuracy: %f' %(acc))

#print (result.shape)

#Plot!!!
#import matplotlib.pyplot as plt
#plt.figure()
#plt.imshow(image[0, :, :, :])
#plt.figure()
#plt.imshow(result[0, :, :, 0])
#plt.figure()
#image[0, :, :, 1] = result[0, :, :, 0] > 0.5
#plt.imshow(image[0, :, :, :])

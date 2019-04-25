import os
import shutil
import random
#import tensorflow as tf
import glob 
import imageio
import numpy as np

#os.remove("/home/vasconcelos/Documents/bichomineirodetectionusingtensorflow (copy)/model.h5")
ImagesTrain = glob.glob('Teste1/Images/*.png')
LabelsTrain = glob.glob('Teste1/Labels/*.png')

ImagesVal = glob.glob('Teste1/Images/*.png')
LabelsVal = glob.glob('Teste1/Labels/*.png')

print(ImagesTrain[37] + ' == '+ LabelsTrain[37])
print('This ^ should be the same.')

dirImage = "Teste1/Images/"
dirLabel = "Teste1/Labels/"

for i, imageFile in enumerate(ImagesTrain):
    
    label = imageio.imread(LabelsTrain[i])

    print(imageFile)
    split = imageFile.split("/")

    
    if 0 in label:
    	print("yes")
    else:
    	os.remove(dirImage + split[2])
    	os.remove(dirLabel + split[2])
    	print(dirImage + split[2])
    	print(dirLabel + split[2])
    	print("no")


'''
i=1;
while i < 100:
	if(os.path.isfile("/home/vasconcelos/Desktop/dados bicho mineiro/train/Images/frame"+str(i)+"-0001.png")):
		#for j in range(5):
		list=[]
		while len(list)<10: 
		#for k in range(5):
			r = random.randint(0,49)
			if r not in list:
				list.append(r)
				
				if len(list) < 6:
					if(len(str(r))==1):
						r = "000"+str(r)
					else:
						r = "00"+str(r)	
					print("image Train: "+str(i+1)+" frame: "+str(r))

					shutil.move("/home/vasconcelos/Desktop/dados bicho mineiro/train/Images/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho mineiro/validation/Images/frame"+str(i)+"-"+str(r)+".png")
					shutil.move("/home/vasconcelos/Desktop/dados bicho mineiro/train/Labels/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho mineiro/validation/Labels/frame"+str(i)+"-"+str(r)+".png")

	i=i+1		
				

		#rand = random.randint(0,50)
		#print("image: "+str(i+1)+ " randon: "+str(j)+" frame: "+str(rand))
		#print(j);
		#print(rand);
		#shutil.move("/home/vasconcelos/Desktop/dados bicho minieor/train/Images/frame1-0000.png", "/home/vasconcelos/Desktop/dados bicho minieor/teste/Images/frame1-0000.png")
'''
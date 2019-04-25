import os
import shutil
import random

#imagens=[1,11,21,34,47,55,67,78,88]
imagens=[3,14,25,33,42,57,63,72,93]
for i in imagens:
	#print(i)
	for j in range(50):
		if(len(str(j))==1):
			j = "000"+str(j)
		else:
			j = "00"+str(j)	
		shutil.move("/home/vasconcelos/Documents/bichomineirodetectionusingtensorflow/TrainSet/Images/"+"frame ("+str(i)+")-"+str(j)+".png","/home/vasconcelos/Documents/bichomineirodetectionusingtensorflow/Teste/Images/"+"frame ("+str(i)+")-"+str(j)+".png")
		shutil.move("/home/vasconcelos/Documents/bichomineirodetectionusingtensorflow/TrainSet/Labels/"+"frame ("+str(i)+")-"+str(j)+".png","/home/vasconcelos/Documents/bichomineirodetectionusingtensorflow/Teste/Labels/"+"frame ("+str(i)+")-"+str(j)+".png")
		
	'''list=[]
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
	'''		
				

		#rand = random.randint(0,50)
		#print("image: "+str(i+1)+ " randon: "+str(j)+" frame: "+str(rand))
		#print(j);
		#print(rand);
		#shutil.move("/home/vasconcelos/Desktop/dados bicho minieor/train/Images/frame1-0000.png", "/home/vasconcelos/Desktop/dados bicho minieor/teste/Images/frame1-0000.png")

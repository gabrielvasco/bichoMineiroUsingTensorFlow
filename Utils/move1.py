import os
import shutil
import random

i=56
for r in range(50):

	if(len(str(r))==1):
		r = "000"+str(r)
	else:
		r = "00"+str(r)	
	print("image Train: "+str(i+1)+" frame: "+str(r))
	shutil.move("/home/vasconcelos/Desktop/dados bicho mineiro/train/Images/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho mineiro/Teste/Images/frame"+str(i)+"-"+str(r)+".png")
	shutil.move("/home/vasconcelos/Desktop/dados bicho mineiro/train/Labels/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho mineiro/Teste/Labels/frame"+str(i)+"-"+str(r)+".png")

'''
i=1;
while i < 80:
	if(os.path.isfile("/home/vasconcelos/Desktop/splittedBichoMineiro/Images/frame"+str(i)+"-0001.png")):
		#for j in range(5):
		list=[]
		while len(list)<10: 
		#for k in range(5):
			r = random.randint(0,60)
			if r not in list:
				list.append(r)
				
				#if len(list) < 6:
				if(len(str(r))==1):
					r = "000"+str(r)
				else:
					r = "00"+str(r)	
				print("image Train: "+str(i+1)+" frame: "+str(r))
				shutil.move("/home/vasconcelos/Desktop/dados bicho mineiro/train/Images/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho minieor/validation/Images/frame"+str(i)+"-"+str(r)+".png")
				shutil.move("/home/vasconcelos/Desktop/dados bicho mineiro/train/Labels/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho minieor/validation/Labels/frame"+str(i)+"-"+str(r)+".png")

				elif len(list) > 5:
					if(len(str(r))==1):
						r = "000"+str(r)
					else:
						r = "00"+str(r)	
					print("image Train: "+str(i+1)+" frame: "+str(r))
					shutil.move("/home/vasconcelos/Desktop/dados bicho minieor/train/Images/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho minieor/teste/Images/frame"+str(i)+"-"+str(r)+".png")
					shutil.move("/home/vasconcelos/Desktop/dados bicho minieor/train/Labels/frame"+str(i)+"-"+str(r)+".png", "/home/vasconcelos/Desktop/dados bicho minieor/teste/Labels/frame"+str(i)+"-"+str(r)+".png")
				'''
	#i=i+1		
				

		#rand = random.randint(0,50)
		#print("image: "+str(i+1)+ " randon: "+str(j)+" frame: "+str(rand))
		#print(j);
		#print(rand);
		#shutil.move("/home/vasconcelos/Desktop/dados bicho minieor/train/Images/frame1-0000.png", "/home/vasconcelos/Desktop/dados bicho minieor/teste/Images/frame1-0000.png")

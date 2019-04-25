from PIL import Image
import os


images = os.listdir("TrainSet/Train/Images/")
path = "TrainSet/Train/Images/"
for image in images:
	print(image)
	Image.open(path+image).convert('RGB').save(path+image)

images = os.listdir("TrainSet/Train/Labels/")
path = "TrainSet/Train/Labels/"
for image in images:
	print(image)
	Image.open(path+image).convert('RGB').save(path+image)

images = os.listdir("Teste/Images/")
path = "Teste/Images/"
for image in images:
	print(image)
	Image.open(path+image).convert('RGB').save(path+image)

images = os.listdir("Teste/Labels/")
path = "Teste/Labels/"
for image in images:
	print(image)
	Image.open(path+image).convert('RGB').save(path+image)
	
images = os.listdir("Validation/Images/")
path = "Validation/Images/"
for image in images:
	print(image)
	Image.open(path+image).convert('RGB').save(path+image)

images = os.listdir("Validation/Labels/")
path = "Validation/Labels/"
for image in images:
	print(image)
	Image.open(path+image).convert('RGB').save(path+image)
		

#for i in range(1207):
#	Image.open("img"+str(i)+".png").convert('RGB').save("img"+str(i)+".png")
	
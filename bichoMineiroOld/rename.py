from PIL import Image

for i in range(1207):
	Image.open("img"+str(i)+".png").convert('RGB').save("img"+str(i)+".png")
	
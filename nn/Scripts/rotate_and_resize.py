from PIL import Image
import os


filenames = []
for root, dirs, files in os.walk(os.path.abspath('.')):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
             filenames.append(root + '\\' + file)
  
  
for filename in filenames:
	im1 = Image.open(filename)  

    width, height = im1.size  
    if height > width:
		im1 = im1.rotate(90, expand=True)

	newsize = (640, 480) 
	im1 = im1.resize(newsize)  
	im1.save(filename) 

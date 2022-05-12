import os



files = []
for file in os.listdir():
	if ' ' in file:
		files.append(file)

for file in files:
	os.rename(file, file.replace(' ', '_'))

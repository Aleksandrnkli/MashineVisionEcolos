import os 
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Creating list of annotations')
	parser.add_argument('--path', default="E:/DataSets/Test_dataset/Annotations",
						help='the annotations path')
	args = parser.parse_args()
	files = [file.replace('.xml', '') + '\n' for file in os.listdir(args.path)]

	with open(args.path + 'files.txt', 'w') as lst_file:
		lst_file.writelines(files)
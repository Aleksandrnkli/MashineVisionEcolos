import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Creating list of annotations')
	parser.add_argument('--path',
									   default="E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class/Annotations",
									   help='the annotations path')
	parser.add_argument('--class_name',
									   default="RU",
									   help='class name RU or KZ')
	args = parser.parse_args()

	images = os.listdir(args.path)

	if args.class_name == "RU":
		CHARS_ru = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
					'A', 'B', 'C', 'E', 'H', 'K', 'M', 'P', 'T', 'X', 'Y', 'O', '-']
	elif args.class_name == "KZ":
		CHARS_kz = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
					'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
					'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-']

	for image in images:
		plate_number = image.replace('.jpg', '')
		for c in plate_number:
			if c not in CHARS:
				os.remove(args.path + '/' +image)
				break
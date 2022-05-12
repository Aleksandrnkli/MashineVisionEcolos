from PIL import Image
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtering photos by resolution")
    parser.add_argument("--path", default="E:\\DataSets\\LicensePlate\\PhotoBaseFull\\ImageSets\\Main\\", help="folder with photos path")
    parser.add_argument("--min_height", default=480, help="minimal height")
    parser.add_argument("--min_width", default=640, help="minimal width")
    args = parser.parse_args()

    path = args.path
    filenames = os.listdir(path)
    trainval = []
    test = []

    with open(path + 'trainval.txt', 'r') as train_txt:
      trainval = [file + '\n' for file in train_txt.readlines()]

    with open(path + 'test.txt', 'r') as test_txt:
      test = [file + '\n' for file in test_txt.readlines()]

    for filename in filenames:
      im1 = Image.open(filename)

      width, height = im1.size
      im1.close()
      if height < args.min_height or width < args.min_width:
        os.remove(filename)

        filename_without_ext = filename.replace('.bmp', '') + '\n'
        if filename_without_ext in trainval:
            trainval.remove(filename_without_ext)

        if filename_without_ext in test:
            test.remove(filename_without_ext)

    with open(path + 'trainval_new.txt', 'w') as train_txt:
        train_txt.writelines(trainval)

    with open(path + 'test_new.txt', 'w') as test_txt:
        test_txt.writelines(test)
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename by convension')
    parser.add_argument('--dataset_path',
                               default="E:/images/Police",
                               help='the annotations path')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    annotations_path = os.path.join(dataset_path, 'Annotations')
    images_path = os.path.join(dataset_path, 'JPEGimages')

    annotations_list = os.listdir(annotations_path)
    images_list = os.listdir(images_path)

    for img in images_list:
        if img.endswith('.bmp'):
            os.rename(images_path + '\\' + img, images_path + '\\' + img.replace('.bmp', '.jpg'))


    symbols_num = 9
    count = 1
    for anno in annotations_list:
        if anno.replace('.xml', '.jpg') not in images_list:
            continue


        new_name = f'{count}'

        while len(new_name) != symbols_num:
            new_name = '0' + new_name

        os.rename(annotations_path + '\\' + anno, annotations_path + '\\' + new_name + '.xml')
        os.rename(images_path + '\\' + anno.replace('.xml', '.jpg'), images_path + '\\' + new_name + '.jpg')

        count = count + 1
import cv2
import os
from crop_lpr_aux_generator import AuxGeneratorForLPRCrop
import argparse

dataset_path = 'E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class'
destination_folder_root = 'C:/Datasets/Cropped'
image_extension = '.jpg'

def convert_generator(generator, destination_folder_root):
    iterations = 0
    for sample in generator:
        iterations += 1
        image_group, annotations_groups = sample
        image = image_group[0]
        annotations = annotations_groups[0]

        # gt_boxes = annotations['boxes']
        # cls_ids = annotations['labels']
        # assert ('boxes' in annotations)
        # assert ('labels' in annotations)
        # assert (annotations['boxes'].shape[0] == annotations['labels'].shape[0])

        for i in range(annotations['boxes'].shape[0]):
            if annotations['labels'][i] == 1:
                destination_folder = os.path.join(destination_folder_root, 'RU')
                # print('ru')
            elif annotations['labels'][i] == 2:
                destination_folder = os.path.join(destination_folder_root, 'KZ')
                # print('kz')
            else:
                destination_folder = os.path.join(destination_folder_root, 'undefined')

            platenum = annotations['platenums'][i][0]
            box = annotations['boxes'][i].astype(int)
            # w = box[2] - box[0] + 1
            # h = box[3] - box[1] + 1
            img_crop = image[box[1]:box[3] + 1, box[0]:box[2] + 1, :]
            # cv2.imshow('', img_crop)
            # cv2.waitKey(3000)
            full_file_name = os.path.join(destination_folder, f'{platenum}.jpg')
            postfix = 0
            while os.path.exists(full_file_name):
                full_file_name = os.path.join(destination_folder, f'{platenum}_{postfix}.jpg')
                postfix += 1

            cv2.imwrite(full_file_name, img_crop)

        print(f'set instance # {iterations}')
        if iterations >= generator.size():
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break


def main(args):
    # create the generators
    image_extension = args.image_extension
    dataset_path = args.dataset_path
    destination_folder_root = args.destination_folder_root

    train_generator = AuxGeneratorForLPRCrop(
        dataset_path,
        'trainval',
        image_extension=image_extension,
        shuffle_groups=False,
        classes={
            "RU plate number" : 1,
            "KZ plate number" : 2,
        }
    )

    validation_generator = AuxGeneratorForLPRCrop(
        dataset_path,
        'test',
        image_extension=image_extension,
        shuffle_groups=False,
        classes={
            "RU plate number": 1,
            "KZ plate number": 2,
        }
    )

    convert_generator(train_generator, destination_folder_root)
    convert_generator(validation_generator, destination_folder_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop license plate images for lpr training")
    parser.add_argument("path", default='E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class', help="path to the datatset")
    parser.add_argument("destination_folder_root", default='E:/Datasets/Cropped', help="destination folder path")
    parser.add_argument("--image_extension", default=".jpg", help="image extension")
    args = parser.parse_args()
    main(args)

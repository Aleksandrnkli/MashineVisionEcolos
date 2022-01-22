import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="remove annotations without photo")
    parser.add_argument("--anno_path", default="E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class/Annotations", help="annotations path")
    parser.add_argument("--images_path", default="E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class/JPEGimages", help="photos path")
    parser.add_argument("--anno_format", default=".xml", help="annotation format")
    parser.add_argument("--photo_format", default=".jpg", help="minimal width")
    args = parser.parse_args()

    anno_format = args.anno_format
    anno_path = args.anno_path
    photo_format = args.photo_format
    images_path = args.images_path

    annotations = [item.replace(anno_format, '') for item in os.listdir(anno_path)]
    photos = [item.replace(photo_format, '') for item in os.listdir(images_path)]

    for photo in photos:
        if photo not in annotations:
            os.remove(images_path + '/' + photo + photo_format)

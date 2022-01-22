import glob
import random
import os
import numpy as np


def annotation_retriever(img_dir):
    img_paths = []
    img_paths += glob.glob(os.path.join(img_dir, '*.jpg'))

    random.shuffle(img_paths)
    num = len(img_paths)
    print("%d pics in total" % num)

    annotations = {}
    for im_path in img_paths:
        basename = os.path.basename(im_path)
        imgname, suffix = os.path.splitext(basename)
        imgname_split = imgname.split('-')
        rec_x1y1 = imgname_split[2].split('_')[0].split('&')
        rec_x2y2 = imgname_split[2].split('_')[1].split('&')
        x1, y1, x2, y2 = int(rec_x1y1[0]), int(rec_x1y1[1]), int(rec_x2y2[0]), int(rec_x2y2[1])

        boxes = np.zeros((1, 4), dtype=np.int32)
        boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3] = x1, y1, x2, y2
        annotations[im_path] = {'boxes': boxes, 'labels': 0}

    print(f'{len(annotations)} annotations have been prepared.')
    return annotations

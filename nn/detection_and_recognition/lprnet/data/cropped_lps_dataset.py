import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import cv2
import os
import glob
from lprnet.preprocessing import lprnet_preprocess

CHARS_ru = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'E', 'H', 'K', 'M', 'P', 'T', 'X', 'Y', 'O', '-']

CHARS_kz = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-']


class LPRDataset(Dataset):
    def __init__(self, img_dir, img_size, class_id, preproc_func=None):
        self.img_dir = img_dir
        self.img_paths = []
        self.class_id = class_id
        for i in range(len(img_dir)):
            self.img_paths += glob.glob(os.path.join(img_dir[i], '*.jpg'))
        random.shuffle(self.img_paths)
        self.img_size = img_size

        if preproc_func is not None:
            self.preproc_func = preproc_func
        else:
            self.preproc_func = lprnet_preprocess

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = cv2.imread(filename)
        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)
        image = self.preproc_func(image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("_")[0]
        label = list()
        if self.class_id == 'RU':
            CHARS = CHARS_ru
        elif self.class_id == 'KZ':
            CHARS = CHARS_kz

        CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

        for c in imgname:
            label.append(CHARS_DICT[c])

        return image, label, len(label)


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


if __name__ == "__main__":

    dataset = LPRDataset(['validation'], (94, 24))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print('data length is {}'.format(len(dataset)))
    for imgs, labels, lengths in dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))      
        break
    

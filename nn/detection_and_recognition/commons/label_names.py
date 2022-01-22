
def custom_label_to_name(label):
    dictionary = {
        0: 'plate number'
    }
    return dictionary[label]


def pascal_voc_label_to_name(label):
    dictionary = {
        0: 'aeroplane',
        1: 'bicycle',
        2: 'bird',
        3: 'boat',
        4: 'bottle',
        5: 'bus',
        6: 'car',
        7: 'cat',
        8: 'chair',
        9: 'cow',
        10: 'diningtable',
        11: 'dog',
        12: 'horse',
        13: 'motorbike',
        14: 'person',
        15: 'pottedplant',
        16: 'sheep',
        17: 'sofa',
        18: 'train',
        19: 'tvmonitor'
    }

    return dictionary[label]


def coco_label_names(label):
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    dictionary = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'street sign',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'hat',
        26: 'backpack',
        27: 'umbrella',
        28: 'shoe',
        29: 'eye glasses',
        30: 'handbag',
        31: 'tie',
        32: 'suitcase',
        33: 'frisbee',
        34: 'skis',
        35: 'snowboard',
        36: 'sports ball',
        37: 'kite',
        38: 'baseball bat',
        39: 'baseball glove',
        40: 'skateboard',
        41: 'surfboard',
        42: 'tennis racket',
        43: 'bottle',
        44: 'plate',
        45: 'wine glass',
        46: 'cup',
        47: 'fork',
        48: 'knife',
        49: 'spoon',
        50: 'bowl',
        51: 'banana',
        52: 'apple',
        53: 'sandwich',
        54: 'orange',
        55: 'broccoli',
        56: 'carrot',
        57: 'hot dog',
        58: 'pizza',
        59: 'donut',
        60: 'cake',
        61: 'chair',
        62: 'couch',
        63: 'potted plant',
        64: 'bed',
        65: 'mirror',
        66: 'dining table',
        67: 'window',
        68: 'desk',
        69: 'toilet',
        70: 'door',
        71: 'tv',
        72: 'laptop',
        73: 'mouse',
        74: 'remote',
        75: 'keyboard',
        76: 'cell phone',
        77: 'microwave',
        78: 'oven',
        79: 'toaster',
        80: 'sink',
        81: 'refrigerator',
        82: 'blender',
        83: 'book',
        84: 'clock',
        85: 'vase',
        86: 'scissors',
        87: 'teddy bear',
        88: 'hair drier',
        89: 'toothbrush',
        90: 'hair brush'
    }

    return dictionary[label]


def get_class_name(cls):
    if cls == 1:
        cls_name = 'firetruck'
    elif cls == 2:
        cls_name = 'police'
    elif cls == 3:
        cls_name = 'ambulance'
    elif cls == 4:
        cls_name = 'car'
    elif cls == 5:
        cls_name = 'bus'
    elif cls == 6:
        cls_name = 'truck'
    else:
        cls_name = " "
    return cls_name


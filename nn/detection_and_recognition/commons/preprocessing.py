import numpy as np


def retina_preprocess(image):
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(float)
    image /= 255.
    return image

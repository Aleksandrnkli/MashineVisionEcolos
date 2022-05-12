import numpy as np


def mtcnn_preprocess(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img[:, :, ::-1]
    img = np.asarray(img, 'float32')
    img = img.transpose((2, 0, 1))
    img = (img - 127.5) * 0.0078125
    return img
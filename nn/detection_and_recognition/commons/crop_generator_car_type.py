from commons.custom_iterator.dataset_access.preprocessing.generator import Generator
from commons.custom_iterator.dataset_access.utils.image import read_image_bgr
import os
import numpy as np
from six import raise_from
# from PIL import Image
import cv2

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


# TODO reconsider necessity to inheritance from Generator. AuxGeneratorForCarModeCrop has very specific purpose that differs from other generators
class AuxGeneratorForCarModeCrop(Generator):

    def __init__(
        self,
        data_dir,
        set_name,
        classes,
        image_extension='.jpg',
        skip_truncated=False,
        skip_difficult=False,
        resize_images=True,
        **kwargs
    ):
        self.data_dir             = data_dir
        self.classes              = classes
        self.set_name             = set_name
        self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult
        self.resize_images        = resize_images

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(AuxGeneratorForCarModeCrop, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        # image = Image.open(path)
        image = cv2.imread(path)

        # return float(image.width) / float(image.height)
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        return read_image_bgr(path)

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def __parse_annotation(self, element):
        """ Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        label = self.name_to_label(class_name)

        box = np.zeros((4,))

        bndbox    = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box.astype(int), label

    def __parse_annotations(self, xml_root):
        """ Parse all annotations under the xml_root.
        """
        # https://www.geeksforgeeks.org/modify-numpy-array-to-store-an-arbitrary-length-string/
        annotations = {'labels': np.empty((len(xml_root.findall('object')),), dtype=int),
                       'boxes': np.empty((len(xml_root.findall('object')), 4), dtype=int)
                       }
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box, label = self.__parse_annotation(element)

                if truncated and self.skip_truncated:
                    continue
                if difficult and self.skip_difficult:
                    continue

                annotations['boxes'][i, :] = box
                annotations['labels'][i] = label
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

        return annotations

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)

    def image_group_paths(self, index):
        # this index corresponds to the passed-in to __get_item__() method
        # and the method is used as workaround to get images filenames
        group = self.groups[index]
        paths = []
        for image_index in group:
            path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
            paths.append(path)

        # return [os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension) for image_index in group]
        return paths

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        if self.resize_images:
            # TODO neither for croping (conversion to LPR's dataset) logic nor for conversion to MTCNN's datatset this resize is not required
            return super(AuxGeneratorForCarModeCrop, self).resize_image(image)
        else:
            return image, 1

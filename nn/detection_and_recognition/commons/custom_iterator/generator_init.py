from .dataset_access.preprocessing.csv_generator import CSVGenerator
from .dataset_access.preprocessing.kitti import KittiGenerator
from .dataset_access.preprocessing.open_images import OpenImagesGenerator
from .dataset_access.preprocessing.pascal_voc import PascalVocGenerator
from .dataset_access.preprocessing.custom_pascal import CustomPascalGenerator
from .dataset_access.utils.transform import random_transform_generator
from .dataset_access.utils.image import random_visual_effect_generator


def create_generators(dataset_type, dataset_path, batch_size=1, random_transform=False, custom_classes=None, image_extension='.jpg', shuffle_groups=False):
    """ Create generators for training and validation.

    Args
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': batch_size
    }

    # for csv
    annotations = ''
    val_annotations = ''
    classes = ''

    # for open image
    version = ''
    labels_filter = ''
    annotation_cache_dir = ''
    parent_label = ''

    # create random transform generator for augmenting training data
    if random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = None
        visual_effect_generator = None

    if dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        # pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
        from .dataset_access.preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            dataset_path,
            'train2017',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            dataset_path,
            'val2017',
            shuffle_groups=shuffle_groups,
            **common_args
        )
    elif dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            dataset_path,
            'trainval',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            dataset_path,
            'test',
            shuffle_groups=shuffle_groups,
            **common_args
        )
    elif dataset_type == 'pascal_custom':
        if custom_classes is None:
            raise ValueError('Custom classes must be provided for custom dataset iterator.')

        train_generator = CustomPascalGenerator(
            dataset_path,
            'trainval',
            classes=custom_classes,
            image_extension=image_extension,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = CustomPascalGenerator(
            dataset_path,
            'test',
            classes=custom_classes,
            image_extension=image_extension,
            shuffle_groups=shuffle_groups,
            **common_args
        )
    elif dataset_type == 'csv':
        train_generator = CSVGenerator(
            annotations,
            classes,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        if val_annotations:
            validation_generator = CSVGenerator(
                val_annotations,
                classes,
                shuffle_groups=shuffle_groups,
                **common_args
            )
        else:
            validation_generator = None
    elif dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            dataset_path,
            subset='train',
            version=version,
            labels_filter=labels_filter,
            annotation_cache_dir=annotation_cache_dir,
            parent_label=parent_label,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = OpenImagesGenerator(
            dataset_path,
            subset='validation',
            version=version,
            labels_filter=labels_filter,
            annotation_cache_dir=annotation_cache_dir,
            parent_label=parent_label,
            shuffle_groups=shuffle_groups,
            **common_args
        )
    elif dataset_type == 'kitti':
        train_generator = KittiGenerator(
            dataset_path,
            subset='train',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = KittiGenerator(
            dataset_path,
            subset='val',
            shuffle_groups=shuffle_groups,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(dataset_type))

    return train_generator, validation_generator


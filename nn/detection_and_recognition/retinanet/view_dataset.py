import cv2
from commons.custom_iterator.generator_init import create_generators
from commons.custom_iterator.dataset_access.utils import draw_annotations
from commons.label_names import custom_label_to_name

batch_size = 1
# for training on COCO
# num_classes = 91  # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# dataset_type = 'coco'
# dataset_path = 'E:/Datasets/COCO/'
# custom_classes = None
# label_to_name = None
# coco_label_name func is not compatible with COCO2017 dataset

# for training on pascal
# num_classes = 19
# dataset_type = 'pascal'
# dataset_path = 'E:/Datasets/Pascal/VOC2012'
# custom_classes = None
# label_to_name = pascal_voc_label_to_name

# for training on custom in pascal format
num_classes = 1
dataset_type = 'pascal_custom'
dataset_path = 'C:/Datasets/Russian_KZ_LPR_detection_2_class'
custom_classes = {
    "RU plate number": 1,
    "KZ plate number": 2,
}
label_to_name = custom_label_to_name


def main():
    # create the generators
    train_generator, validation_generator = create_generators(dataset_type=dataset_type, dataset_path=dataset_path, batch_size=batch_size, custom_classes=custom_classes)

    # simply provoke iterations through the data
    batches = 0
    for sample in train_generator:
        batches += 1
        image_group, annotations_groups = sample
        for i in range(batch_size):
            image = image_group[i]
            annotations = annotations_groups[i]
            # gt_boxes = annotations['boxes']
            # cls_ids = annotations['labels']
            draw_annotations(image, annotations, label_to_name=label_to_name)
            cv2.imshow('dataset viewer', image)

            if cv2.waitKey(3000) & 0xFF == ord('q'):
                return

        print(f'batch # {batches}')
        if batches >= train_generator.size() // batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break


if __name__ == '__main__':
    main()

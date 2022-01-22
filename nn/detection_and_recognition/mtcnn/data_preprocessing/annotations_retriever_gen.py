from commons.crop_lpr_aux_generator import AuxGeneratorForLPRCrop
from commons.crop_generator_car_type import AuxGeneratorForCarModeCrop

def annotation_retriever(learning_mode, dataset_path, dataset_part='trainval'):
    print('starting to convert the dataset for MTCNN...')
    result_annotations = {}

    image_extension = '.jpg'

    # indexes started from 1, because 0 and -1 indexes used as background and ignoring labels respectively
    if learning_mode =="LPR":
        custom_classes = {
            'RU plate number': 1,
            'KZ plate number': 2
        }
        generator = AuxGeneratorForLPRCrop(dataset_path, dataset_part, classes=custom_classes,
                                           image_extension=image_extension,shuffle_groups=False , batch_size=1,
                                           resize_images=False)

    elif learning_mode == "CAR":
        custom_classes = {
            'firetruck': 1,
            'police': 2,
            'ambulance': 3,
            'car': 4,
            'bus': 5,
            'truck': 6,
        }
        generator = AuxGeneratorForCarModeCrop(dataset_path, dataset_part, classes=custom_classes, image_extension=image_extension, shuffle_groups=False, batch_size=1,
                                           resize_images=False)


    iterations = 0
    for sample in generator:
        image_group, annotations_groups = sample
        # image = image_group[0]
        annotations = annotations_groups[0]

        # gt_boxes = annotations['boxes']
        # cls_ids = annotations['labels']
        # nums = annotations['platenums'] only for LPR mode
        # assert ('boxes' in annotations)
        # assert ('labels' in annotations)
        # assert ('nums' in annotations) only for LPR mode
        # assert (annotations['boxes'].shape[0] == annotations['labels'].shape[0]) == annotations['platenums'].shape[0]) For LPR mode
        # assert (annotations['boxes'].shape[0] == annotations['labels'].shape[0]) For CAR mode
        image_paths = generator.image_group_paths(iterations)
        image_path = image_paths[0]

        result_annotations[image_path] = annotations

        print(f'dataset instance # {iterations}')
        iterations += 1
        if iterations >= generator.size():
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

    return result_annotations

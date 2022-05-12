import cv2
from commons.custom_iterator.generator_init import create_generators
from commons.custom_iterator.dataset_access.utils import draw_detections
import torch
import torchvision
import os
from commons.label_names import coco_label_names
from commons.preprocessing import retina_preprocess

batch_size = 1
epoch_to_use = 5  # epoch to use in demo

# for training on COCO
use_pretrained = True

num_classes = 91  # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
dataset_type = 'coco'
dataset_path = 'E:/Datasets/COCO/'
custom_classes = None
label_to_name = coco_label_names

# for training on pascal
# num_classes = 19
# dataset_type = 'pascal'
# dataset_path = 'E:/Datasets/Pascal/VOC2012'
# custom_classes = None
# label_to_name = pascal_voc_label_to_name

# for training on custom in pascal format
# num_classes = 1
# dataset_type = 'pascal_custom'
# dataset_path = 'D:/Datasets/Lastochka'
# custom_classes = {
#     'floor wash machine': 0,
# }
# label_to_name = custom_label_to_name


def main():
    # create the generators
    train_generator, validation_generator = create_generators(dataset_type=dataset_type, dataset_path=dataset_path, batch_size=batch_size, custom_classes=custom_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_pretrained:
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    else:
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=num_classes)
    model.eval()
    model.to(device)

    if not use_pretrained:
        checkpoint_path = f'./{dataset_type}_weights/retinanet_{epoch_to_use}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])  #, strict=False
            print(f'weights {checkpoint_path} has been loaded')

    batches = 0
    for sample in train_generator:
        batches += 1
        image_group, _ = sample

        image_group_on_device = []
        for image in image_group:

            image = retina_preprocess(image)

            image_group_on_device.append(torch.tensor(image, dtype=torch.float32).to(device))

        detections_group = model(image_group_on_device)

        detections_group_off_device = []
        for detections in detections_group:
            detections_group_off_device.append({'boxes': detections['boxes'].detach().cpu(), 'scores': detections['scores'].detach().cpu(), 'cls': detections['labels'].detach().cpu()})

        for i in range(batch_size):
            image = image_group[i]
            detections = detections_group_off_device[i]
            draw_detections(image, detections['boxes'].numpy(), detections['scores'].numpy(), detections['cls'].numpy(), label_to_name=label_to_name)  # , score_threshold=0.9
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

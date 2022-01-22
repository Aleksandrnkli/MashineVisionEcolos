import cv2
from commons.custom_iterator.dataset_access.utils import draw_detections
import torch
import torchvision
import os
from commons.label_names import custom_label_to_name
from commons.preprocessing import retina_preprocess

epoch_to_use = 14  # epoch to use in demo

# for training on COCO
# num_classes = 91  # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# dataset_type = 'coco'
# custom_classes = None
# label_to_name = coco_label_names

# for training on pascal
# num_classes = 19
# dataset_type = 'pascal'
# custom_classes = None
# label_to_name = pascal_voc_label_to_name

# for training on custom in pascal format
num_classes = 1
dataset_type = 'pascal_custom'
custom_classes = {
    'floor wash machine': 0,
}
label_to_name = custom_label_to_name


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=num_classes)
    model.eval()
    model.to(device)

    checkpoint_path = f'./{dataset_type}_weights/retinanet_{epoch_to_use}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])  #, strict=False
        print(f'weights {checkpoint_path} has been loaded')

    # file_full_name = 'E:/DataSets/Lastochka/Lastochka-Test-Videos/Disinfection/643-2020-11-04_20.46.01_00039_ дезинфекция.mp4'
    # file_full_name = 'E:/DataSets/Lastochka/Lastochka-Test-Videos/Disinfection/669-2020-11-04_20.46.03_00039_по_669-2020-11-04_20.46.03_00041_дезинфекция.mp4'
    # file_full_name = 'E:/DataSets/Lastochka/Lastochka-Test-Videos/Disinfection/643-2020-11-04_20.46.04_00045_дизенфекция.mp4'
    file_full_name = 'E:/DataSets/Lastochka/Lastochka-Test-Videos/WashShop_WashMachine/Wash_mach_3_for_demo.mp4'

    video_capture = cv2.VideoCapture(file_full_name)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        image_group_on_device = []
        image = retina_preprocess(frame)

        image_group_on_device.append(torch.tensor(image, dtype=torch.float).to(device))

        detections_group = model(image_group_on_device)

        detections = detections_group[0]
        detections_off_device = {'boxes': detections['boxes'].detach().cpu(), 'scores': detections['scores'].detach().cpu(), 'cls': detections['labels'].detach().cpu()}

        draw_detections(frame, detections_off_device['boxes'].numpy(), detections_off_device['scores'].numpy(), detections_off_device['cls'].numpy(), score_threshold=0.9, label_to_name=label_to_name)
        cv2.imshow('dataset viewer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

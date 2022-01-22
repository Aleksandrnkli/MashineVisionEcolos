import cv2
import numpy as np
import torch
import torchvision
from commons.label_names import coco_label_names
from commons.custom_iterator.dataset_access.utils import draw_detections
import datetime


def has_equal_or_close_box(prev_boxes, curr_box):
    """
    :param prev_boxes: list of lists, reference boxes of previous save iteration
    :param curr_box: list, one of the current boxes
    """

    scale_factor = 0  # add later the logic of the dependence between acceptable distance and resolution
    acceptable_distance = 600

    if len(prev_boxes) == 0:
        return False

    for prev_box in prev_boxes:
        x1 = True if abs(curr_box[0] - prev_box[0]) <= acceptable_distance else False
        y1 = True if abs(curr_box[1] - prev_box[1]) <= acceptable_distance else False
        x2 = True if abs(curr_box[2] - prev_box[2]) <= acceptable_distance else False
        y2 = True if abs(curr_box[3] - prev_box[3]) <= acceptable_distance else False
        if x1 and y1 and x2 and y2:
            return True

    return False


label_to_name = coco_label_names

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

video_capture = cv2.VideoCapture('rtsp://admin:admin@10.1.2.224:554')

# next_second = 0
# the bounding boxes of the previous image save iteration
saved_boxes = []
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    image = np.transpose(frame, (2, 0, 1))
    image = image.astype(float)
    image /= 255.

    x = [torch.tensor(image, dtype=torch.float).to(device)]

    detections_group = model(x)

    detections = detections_group[0]
    detections_off_device = {'boxes': detections['boxes'].detach().cpu(), 'scores': detections['scores'].detach().cpu(),
                             'cls': detections['labels'].detach().cpu()}

    score_threshold = 0.5
    scores = detections_off_device['scores'].numpy()
    boxes = detections_off_device['boxes'].numpy()
    labels = detections_off_device['cls'].numpy()
    selection = np.where(scores > score_threshold)[0]
    for i in selection:
        if labels[i] == 3:  # - the cars class in terms of the Retina, but class #2 in mapping!
            box = list(boxes[i, :])
            if not has_equal_or_close_box(saved_boxes, box):
                dt = datetime.datetime.now()
                timestamp = f'{dt.day}-{dt.month}-{dt.year}_{dt.hour}-{dt.minute}-{dt.second}'
                cv2.imwrite('./images/' + timestamp + '.png', frame)
                saved_boxes.clear()
                for j in selection:
                    saved_boxes.append(box)
                break

        # if list(boxes[i, :]) not in saved_boxes:
        #     dt = datetime.datetime.now()
        #     timestamp = f'{dt.day}-{dt.month}-{dt.year}_{dt.hour}-{dt.minute}-{dt.second}'
        #     cv2.imwrite('./images/' + timestamp + '.png', frame)
        #     saved_boxes.clear()
        #     for i in selection:
        #         saved_boxes.append(list(boxes[i, :]))
        #     break

        # contain_objects = False
        # for score in detections_off_device['scores']:
        #     if score >= 0.5:
        #         contain_objects = True
        #         break
        #
        # dt = datetime.datetime.now()
        # if dt.second == next_second and contain_objects:
        #     timestamp = f'{dt.day}-{dt.month}-{dt.year}_{dt.hour}-{dt.minute}-{dt.second}'
        #     cv2.imwrite('./images/' + timestamp + '.png', frame)
        #     if dt.second == 45:
        #         next_second = 0
        #     else:
        #         next_second += 15

    draw_detections(frame, detections_off_device['boxes'].numpy(), detections_off_device['scores'].numpy(),
                    detections_off_device['cls'].numpy(), label_to_name=label_to_name)  # , score_threshold=0.9
    cv2.imshow('dataset viewer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

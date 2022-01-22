import cv2
import numpy as np
import torch
import torchvision
from commons.label_names import custom_label_to_name
from commons.custom_iterator.dataset_access.utils import draw_detections
import datetime
import os


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


epoch_to_use = 100
num_classes = 1
dataset_type = 'pascal_custom'
custom_classes = {
    'plate number': 0,
}

label_to_name = custom_label_to_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=num_classes)
model.eval()
model.to(device)

checkpoint_path = f'./{dataset_type}_weights/retinanet_{epoch_to_use}.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # , strict=False
    print(f'weights {checkpoint_path} has been loaded')

video_capture = cv2.VideoCapture('rtsp://admin:admin@10.1.2.224:554')

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
    cls = detections_off_device['labels'].numpy()
    selection = np.where(scores > score_threshold)[0]
    for i in selection:
        box = list(boxes[i, :])
        box_dict = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
        is_present = False
        for saved_box in saved_boxes:
            saved_dict = {'x1': saved_box[0], 'y1': saved_box[1], 'x2': saved_box[2], 'y2': saved_box[3]}
            intersection_percent = get_iou(box_dict, saved_dict)
            if intersection_percent > 0.3:
                is_present = True
        if not is_present:
            dt = datetime.datetime.now()
            timestamp = f'{dt.day}-{dt.month}-{dt.year}_{dt.hour}-{dt.minute}-{dt.second}'
            cv2.imwrite('./images/' + timestamp + '.png', frame)
            saved_boxes.clear()
            for j in selection:
                saved_boxes.append(list(boxes[j, :]))
            break

    draw_detections(frame, detections_off_device['boxes'].numpy(), detections_off_device['scores'].numpy(),
                    detections_off_device['cls'].numpy(), label_to_name=label_to_name)  # , score_threshold=0.9
    cv2.imshow('viewer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

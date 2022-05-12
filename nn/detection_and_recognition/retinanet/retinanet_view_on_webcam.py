import cv2
import torch
import torchvision
from commons.label_names import coco_label_names
from commons.custom_iterator.dataset_access.utils.draw_detections import draw_detections
from commons.preprocessing import retina_preprocess


label_to_name = coco_label_names

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        break

    # frame = cv2.resize(frame, (img_width, img_height))
    image = retina_preprocess(frame)

    x = [torch.tensor(image, dtype=torch.float).to(device)]

    detections_group = model(x)

    detections = detections_group[0]
    detections_off_device = {'boxes': detections['boxes'].detach().cpu(), 'scores': detections['scores'].detach().cpu(),
                             'cls': detections['labels'].detach().cpu()}

    draw_detections(frame, detections_off_device['boxes'].numpy(), detections_off_device['scores'].numpy(),
                    detections_off_device['cls'].numpy(), label_to_name=label_to_name)  # , score_threshold=0.9

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

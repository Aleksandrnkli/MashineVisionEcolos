import numpy as np
import cv2



def draw_detections(image, car_boxes, car_scores, car_cls,
                    boxes, scores, labels, color=(0, 255, 0), score_threshold=0.5):
    selection = np.where(scores > score_threshold)[0]
    thickness = 2
    for i in selection:
        b = boxes[i, :]
        b = np.array(b).astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

        caption = str(labels[i][0])  # + ': {0:.2f}'.format(scores[i])
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    car_selection = np.where(car_scores > score_threshold)[0]
    thickness = 2
    for i in car_selection:
        b2 = car_boxes[i, :]
        b2= np.array(b2).astype(int)
        cv2.rectangle(image, (b2[0], b2[1]), (b2[2], b2[3]), color, thickness, cv2.LINE_AA)

        caption = str(car_cls)
        cv2.putText(image, caption, (b2[0], b2[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b2[0], b2[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)



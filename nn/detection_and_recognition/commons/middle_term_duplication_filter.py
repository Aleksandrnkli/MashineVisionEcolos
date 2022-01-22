from commons.indexed_priority_queue import IndexedMinPQ
import datetime
import numpy as np


class MiddleTermDuplicationFilter:
    def __init__(self, time_threshold, score_threshold):
        self.time_threshold = time_threshold
        self.score_threshold = score_threshold
        self.ipq = IndexedMinPQ(500)

    def do_filter(self, images, detections_batch):
        # update the queue leaving only the number which aren't lost yet
        while not self.ipq.is_empty() and (datetime.datetime.now() - self.ipq.peek()[1]).total_seconds() > self.time_threshold:
            # dequeue the number which out of time range
            _, _ = self.ipq.del_min()

        best_images_batch = []
        best_detections_batch = []
        best_boxes = []
        best_scores = []
        best_numbers = []
        best_timestamps = []

        #  for each image in the batch
        for image, detections in zip(images, detections_batch):
            detection_on_image = False
            # for each detected number on the image
            boxes, scores, numbers = detections['boxes'], detections['scores'], detections['numbers']
            selection = np.where(scores > self.score_threshold)[0]
            for i in selection:
                box = boxes[i, :]
                box = np.array(box).astype(int)
                score = scores[i]
                number = numbers[i][:]

                # check if the number has been already seen
                if not self.ipq.contains(number):
                    # if not put it in the queue
                    if self.ipq.is_full():
                        ValueError('The queue is full consider to enlarge its size.')
                    timestamp = datetime.datetime.now()
                    self.ipq.insert(number, timestamp)

                    if not detection_on_image:
                        best_images_batch.append(image)
                        detection_on_image = True

                    # and pass-through the filter
                    best_boxes.append(box)
                    best_scores.append(score)
                    best_numbers.append(number)
                    best_timestamps.append(timestamp)
                # else:
                    # if the number has been already seen - filter it out
                    # continue

            if detection_on_image:
                best_detections =   {
                                        'boxes': np.asarray(best_boxes),
                                        'scores': np.asarray(best_scores),
                                        'numbers': np.asarray(best_numbers),
                                        'timestamps': np.asarray(best_timestamps)
                                    }
                best_detections_batch.append(best_detections)

            # best_images_batch = np.asarray(best_images_batch)
        return best_images_batch, best_detections_batch

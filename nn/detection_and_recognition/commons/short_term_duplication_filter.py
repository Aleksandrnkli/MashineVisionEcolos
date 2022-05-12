from commons.indexed_priority_queue import IndexedMinPQ
import datetime
import numpy as np
import re


class ShortTermDuplicationFilter:
    def __init__(self, time_threshold, score_threshold, amount_threshold):
        self.time_threshold = time_threshold
        self.score_threshold = score_threshold
        self.amount_threshold = amount_threshold
        self.hash_table = {}
        self.ipq = IndexedMinPQ(50)

    def do_filter(self, image, detections):
        # for each detected number
        boxes, scores, numbers = detections['boxes'], detections['scores'], detections['numbers']
        selection = np.where(scores > self.score_threshold)[0]
        for i in selection:
            box = boxes[i, :]
            box = np.array(box).astype(int)
            score = scores[i]
            number = numbers[i][0]

            # Unless we are using chars alphabet for decoding there is no
            # way we can bump into Z or something
            kz_regex = r'\d{3}[a-zA-Z]{3}\d{2}'
            ru_regex = r'[a-zA-Z]\d{3}[a-zA-Z]{2}\d{2,3}'
            if not re.fullmatch(ru_regex, number) and not re.fullmatch(kz_regex, number):
                continue

            # put detection time in priority queue
            if self.ipq.is_full():
                ValueError('The queue is full consider to enlarge its size.')
            if not self.ipq.contains(number):
                timestamp = datetime.datetime.now()
                self.ipq.insert(number, timestamp)

            # collect image frames and annotations per number in hash table
            if number not in self.hash_table:
                self.hash_table[number] = list()
            self.hash_table[number].append((image, box, score))

        # if most recent item was stored long ago enough  dequeue the item
        # https://stackoverflow.com/questions/4362491/how-do-i-check-the-difference-in-seconds-between-two-dates

        best_boxes = []
        best_scores = []
        best_images = []
        best_numbers = []
        best_timestamps = []

        best_detections = []
        while not self.ipq.is_empty() and (datetime.datetime.now() - self.ipq.peek()[1]).total_seconds() > self.time_threshold:
            # dequeue the number
            number, timestamp = self.ipq.del_min()

            hash_table_number_entry = self.hash_table[number]  # save the entry on local scope var
            # remove the number from hash
            del self.hash_table[number]

            # for the given number select the 'best' (given chosen metric) frame stored so far
            min_center_offset = 1000000000000  # just a huge constant
            index = -1

            if self.amount_threshold > 0 and len(hash_table_number_entry) < self.amount_threshold:
                continue

            # TODO refactor into numpy logic
            for i, (image, box, score) in enumerate(hash_table_number_entry):
                img_height, img_width, _ = image.shape
                # score may or may not be used in the metric calc as all the numbers in the detection group are the same
                center_offset = (box[0] - img_width / 2) + (box[1] - img_height / 1.5) + (box[2] - img_width / 2) + (
                            box[3] - img_height / 1.5)
                if 0 < center_offset < min_center_offset:
                    min_center_offset = center_offset
                    index = i

            best_images.append(hash_table_number_entry[index][0])

            best_boxes.append(hash_table_number_entry[index][1])
            best_scores.append(hash_table_number_entry[index][2])
            best_numbers.append(number)
            best_timestamps.append(timestamp)

            detection = {'boxes': np.asarray(best_boxes),
                         'scores': np.asarray(best_scores),
                         'timestamps': np.asarray(best_timestamps),
                         'numbers': np.asarray(best_numbers)
                         }

            best_detections.append(detection)

        return best_images, best_detections

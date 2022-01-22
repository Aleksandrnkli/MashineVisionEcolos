import cv2
import torch
from lpr_pipeline import Pipeline
from commons.short_term_duplication_filter import ShortTermDuplicationFilter
from commons.middle_term_duplication_filter import MiddleTermDuplicationFilter
from commons.custom_iterator.dataset_access.utils.draw_detections import draw_detections


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score_threshold = 0.5
    num_classes = 2

    # for usage RetinaNet in the pipeline
    # retina_epoch_to_use = 100  # retina's epoch to use in demo
    # dataset_type = 'pascal_custom'
    # retinanet_checkpoint_path = f'./retinanet/{dataset_type}_weights/retinanet_{retina_epoch_to_use}.pth'

    stnetru_checkpoint_path = 'C:/Models/LPR/RU/stn_Iter_045300_model.ckpt'
    stnetkz_checkpoint_path = 'C:/Models/LPR/KZ/stn_Iter_063000_model.ckpt'
    lprnetru_checkpoint_path = 'C:/Models/LPR/RU/lprnet_Iter_045300_model.ckpt'
    lprnetkz_checkpoint_path = 'C:/Models/LPR/KZ/lprnet_Iter_063000_model.ckpt'
    onet_checkpoint_path = 'C:/Models/LPR/onet_Weights'
    pnet_checkpoint_path = 'C:/Models/LPR/pnet_Weights'

    pipeline = Pipeline(device, num_classes, pnet_checkpoint_path=pnet_checkpoint_path, onet_checkpoint_path=onet_checkpoint_path, stnetru_checkpoint_path=stnetru_checkpoint_path, stnetkz_checkpoint_path=stnetkz_checkpoint_path, lprnetru_checkpoint_path=lprnetru_checkpoint_path, lprnetkz_checkpoint_path=lprnetkz_checkpoint_path)

    video_source_name = 'C:/Datasets/Auto.mkv'
    # video_source_name = 'rtsp://admin:admin@10.1.2.224:554'

    video_capture = cv2.VideoCapture(video_source_name)

    short_term_filter = ShortTermDuplicationFilter(3, 0.5, 3)
    middle_term_filter = MiddleTermDuplicationFilter(900, 0.5)

    while True:
        # capturing frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # pipeline usage
        # TODO now the LPR Pipeline execute() method recieves and returns only one image not a batch
        frame, detections_off_device = pipeline.execute(frame)

        # demo
        # draw_detections(frame, detections_off_device['boxes'], detections_off_device['scores'], detections_off_device['numbers'], score_threshold=score_threshold)
        # cv2.imshow('dataset viewer', frame)
        frames_batch, filtered_detections_batch = short_term_filter.do_filter(frame, detections_off_device)
        if len(frames_batch) == 0:  # it's possible to filter-out all entries so the check is required
            continue

        frames_batch, filtered_detections_batch = middle_term_filter.do_filter(frames_batch, filtered_detections_batch)
        if len(frames_batch) == 0:
            continue

        for frame, detection_filtered in zip(frames_batch, filtered_detections_batch):
            draw_detections(frame, detection_filtered['boxes'], detection_filtered['scores'], detection_filtered['numbers'], score_threshold=score_threshold)
            cv2.imshow('dataset viewer', frame)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # releasing the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

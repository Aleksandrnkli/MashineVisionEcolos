import cv2
import torch
import requests
from lpr_pipeline import Pipeline
from commons.short_term_duplication_filter import ShortTermDuplicationFilter
from commons.middle_term_duplication_filter import MiddleTermDuplicationFilter
from commons.custom_iterator.dataset_access.utils.draw_detections import draw_detections
import argparse


def post_detection(image, detection):
    auth_token = '6ae8ee0fa8c5ac2012e1dfa45d9527fd10ce24e8'
    server_url = "http://10.1.2.187:9999/api/lpr/"
    auth_header = {'Authorization': 'Token ' + auth_token}

    for i in range(len(detection['boxes'])):
        cv2.imwrite('img.jpg', image)
        files = {
            'image': ('image.jpg', cv2.imencode('.jpg', image)[1].tobytes(), 'image/jpeg')
        }

        data = {
            'license_plate': detection['numbers'][i],
            'detection_time': detection['timestamps'][i]
        }
        r = requests.post(server_url, files=files, data=data, headers=auth_header)
        if r.status_code == 201:
            print(f'UPLOADED TO SERVER: {detection["numbers"][i]} at {detection["timestamps"][i]}')
        else:
            print(r.text)


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score_threshold = 0.5
    num_classes = 2

    # for usage RetinaNet in the pipeline
    # retina_epoch_to_use = 100  # retina's epoch to use in demo
    # dataset_type = 'pascal_custom'
    # retinanet_checkpoint_path = f'./retinanet/{dataset_type}_weights/retinanet_{retina_epoch_to_use}.pth'

    stnetru_checkpoint_path = args.stnetru_checkpoint_path
    stnetkz_checkpoint_path = args.stnetkz_checkpoint_path
    lprnetru_checkpoint_path = args.lprnetru_checkpoint_path
    lprnetkz_checkpoint_path = args.lprnetkz_checkpoint_path
    onet_checkpoint_path = args.onet_checkpoint_path
    pnet_checkpoint_path = args.pnet_checkpoint_path

    pipeline = Pipeline(device, num_classes, pnet_checkpoint_path=pnet_checkpoint_path, onet_checkpoint_path=onet_checkpoint_path, stnetru_checkpoint_path=stnetru_checkpoint_path, stnetkz_checkpoint_path=stnetkz_checkpoint_path, lprnetru_checkpoint_path=lprnetru_checkpoint_path, lprnetkz_checkpoint_path=lprnetkz_checkpoint_path)

    video_source_name = args.file_full_name
    # video_source_name = 'C:/Datasets/Auto.mkv'

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
        draw_detections(frame, detections_off_device['boxes'], detections_off_device['scores'], detections_off_device['numbers'], score_threshold=score_threshold)
        cv2.imshow('dataset viewer', frame)

        frames_batch, filtered_detections_batch = short_term_filter.do_filter(frame, detections_off_device)
        if len(frames_batch) == 0:  # it's possible to filter-out all entries so the check is required
            continue

        frames_batch, filtered_detections_batch = middle_term_filter.do_filter(frames_batch, filtered_detections_batch)
        if len(frames_batch) == 0:
            continue

        for frame, detection_filtered in zip(frames_batch, filtered_detections_batch):
            post_detection(frame, detection_filtered)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            if cv2.waitKey(3000) & 0xFF == ord('q'):
                break

    # releasing the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="demo pipeline with lpr detection and lpr recognition with posting on the web part")
    parser.add_argument("--stnetru_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/RU/stn_Iter_045300_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--stnetkz_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/KZ/stn_Iter_063000_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--lprnetru_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/RU/lprnet_Iter_045300_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--lprnetkz_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/KZ/lprnet_Iter_063000_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--onet_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/onet_Weights", help="path to weights for ru_stn")
    parser.add_argument("--pnet_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/pnet_Weights", help="path to weights for ru_stn")
    parser.add_argument("--file_full_name", default="rtsp://admin:admin@10.1.2.224:554/1", help="path to the video file")
    args = parser.parse_args()
    main(args)

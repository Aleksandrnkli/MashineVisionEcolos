import cv2
import numpy as np
import torch
from lpr_pipeline_ext import Pipeline
from commons.custom_iterator.dataset_access.utils.draw_detections import draw_detections
import argparse

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score_threshold = 0.5

    # retina_epoch_to_use = 100  # retina's epoch to use in demo
    num_classes_lpr = 2
    num_classes_car_mode = 6
    # dataset_type = 'pascal_custom'
    stnetru_checkpoint_path = args.stnetru_checkpoint_path
    stnetkz_checkpoint_path = args.stnetkz_checkpoint_path
    lprnetru_checkpoint_path = args.lprnetru_checkpoint_path
    lprnetkz_checkpoint_path = args.lprnetkz_checkpoint_path
    onet_checkpoint_path = args.onet_checkpoint_path
    pnet_checkpoint_path = args.pnet_checkpoint_path
    pnet_car_checkpoint_path = args.pnet_car_checkpoint_path
    onet_car_checkpoint_path = args.onet_car_checkpoint_path
    file_full_name = args.file_full_name

    pipeline = Pipeline(device, num_classes_lpr, num_classes_car_mode, pnet_lpr_checkpoint_path=pnet_checkpoint_path,
                        onet_lpr_checkpoint_path=onet_checkpoint_path, pnet_car_mode_checkpoint_path=pnet_car_checkpoint_path,
                        onet_car_mode_checkpoint_path=onet_car_checkpoint_path, stnetru_checkpoint_path=stnetru_checkpoint_path,
                        stnetkz_checkpoint_path=stnetkz_checkpoint_path, lprnetru_checkpoint_path=lprnetru_checkpoint_path,
                        lprnetkz_checkpoint_path=lprnetkz_checkpoint_path)

    # file_full_name = 'D:/PyCharmProjects/LPR_Pipeline/Auto.mkv'
    # KZ_video1 = 'D:/Share/videoplayback (online-video-cutter.com).mp4'
    # KZ_video2 = 'D:/Share/Daughter_auto.mp4'
    # RTSP_camera = 'rtsp://admin:admin@10.1.2.224:554'

    video_capture = cv2.VideoCapture(file_full_name)

    while True:
        # capturing frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # pipeline usage
        frame, detections_off_device = pipeline.execute(frame)

        # demo
        draw_detections(frame,
                        detections_off_device['car_boxes'], detections_off_device['car_type_scores'], detections_off_device['car_type'],
                        detections_off_device['boxes'], detections_off_device['scores'], detections_off_device['numbers'],
                        score_threshold=score_threshold)
        cv2.imshow('dataset viewer', frame)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            if cv2.waitKey(3000) & 0xFF == ord('q'):
                break

    # releasing the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="demo pipeline with lpr detection and lpr recognition on the test video")
    parser.add_argument("--stnetru_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/RU/stn_Iter_045300_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--stnetkz_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/KZ/stn_Iter_063000_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--lprnetru_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/RU/lprnet_Iter_045300_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--lprnetkz_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/KZ/lprnet_Iter_063000_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--onet_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/onet_Weights", help="path to weights for mtcnn_lpr")
    parser.add_argument("--pnet_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/pnet_Weights", help="path to weights for mtcnn_lpr")
    parser.add_argument("--onet_car_checkpoint_path", default="E:/Models/LPR/car_det_mtcnn/onet_Weights", help="path to weights for mtcnn_car")
    parser.add_argument("--pnet_car_checkpoint_path", default="E:/Models/LPR/car_det_mtcnn/pnet_Weights", help="path to weights for mtcnn_car")
    parser.add_argument("--file_full_name", default="D:/Share/pexels-kelly-lacy-5473765.mp4", help="path to the video file")
    args = parser.parse_args()
    main(args)

import cv2
import numpy as np
import torch
import torchvision
from lprnet.model.lprnet import LPRNet
from lprnet.model.stn import STNet
from lprnet.data.cropped_lps_dataset import CHARS_ru, CHARS_kz
from lprnet.decoders import beam_search_decoder
from mtcnn.model.MTCNN import create_mtcnn_net, execute_mtcnn_net
import os
from commons.preprocessing import retina_preprocess
from lprnet.preprocessing import lprnet_preprocess
from settings import get_const_for_MTCNN
from commons.label_names import get_class_name


class Pipeline:
    def __init__(self, device, num_classes_lpr, num_classes_car_mode, pnet_lpr_checkpoint_path=None, onet_lpr_checkpoint_path=None,
                 pnet_car_mode_checkpoint_path=None, onet_car_mode_checkpoint_path=None, retinanet_checkpoint_path=None,
                 stnetru_checkpoint_path=None, stnetkz_checkpoint_path=None, lprnetru_checkpoint_path=None,
                 lprnetkz_checkpoint_path=None):
        self.device = device

        # pipeline initialization
        if (pnet_lpr_checkpoint_path or onet_lpr_checkpoint_path) and retinanet_checkpoint_path:
            raise ValueError("Only one detector (either RetinaNet or MTCNN (Pnet, Onet)) can be used in the pipeline.")
        if not (pnet_lpr_checkpoint_path and onet_lpr_checkpoint_path) and not retinanet_checkpoint_path:
            raise ValueError("Both Pnet, Onet paths should be provided.")

        self.detector_lp = None

        # mtcnn initialization
        if pnet_lpr_checkpoint_path and onet_lpr_checkpoint_path:
            _, _, _, _, _,kernel, mp, linear = get_const_for_MTCNN(learning_mode="LPR")
            if os.path.exists(pnet_lpr_checkpoint_path) and os.path.exists(onet_lpr_checkpoint_path):
                self.pnet_lp, self.onet_lp = create_mtcnn_net(linear=linear, kernel=kernel, mp=mp, device=device, cls_num=num_classes_lpr, p_model_path=pnet_lpr_checkpoint_path, o_model_path=onet_lpr_checkpoint_path)
                print(f' MTCNN weights {pnet_lpr_checkpoint_path} and {onet_lpr_checkpoint_path} has been loaded')

                self.detector_lp = 'mtcnn_lpr'

        if pnet_car_mode_checkpoint_path and onet_car_mode_checkpoint_path:
            _, _, _, _, _, kernel, mp, linear = get_const_for_MTCNN(learning_mode="CAR")
            if os.path.exists(pnet_car_mode_checkpoint_path) and os.path.exists(onet_car_mode_checkpoint_path):
                self.pnet_car, self.onet_car = create_mtcnn_net(linear=linear, kernel=kernel, mp=mp, device=device, cls_num=num_classes_car_mode, p_model_path=pnet_car_mode_checkpoint_path, o_model_path=onet_car_mode_checkpoint_path)
                print(f' MTCNN weights {pnet_car_mode_checkpoint_path} and {onet_car_mode_checkpoint_path} has been loaded')

                self.detector_car_mode = 'mtcnn_car_mode'

        # retinanet initialization
        if retinanet_checkpoint_path:
            self.retinanet = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=num_classes_lpr)
            self.retinanet.eval()
            self.retinanet.to(device)
            if os.path.exists(retinanet_checkpoint_path):
                checkpoint = torch.load(retinanet_checkpoint_path)
                self.retinanet.load_state_dict(checkpoint['model_state_dict'])  # , strict=False
                print(f' Retinanet weights {retinanet_checkpoint_path} has been loaded')

                self.detector_lp = 'retinanet'
            else:
                raise ValueError(f'Path {retinanet_checkpoint_path} is not found.')

        # spatial transformation net initialization for RU
        self.STNru = STNet()
        self.STNru.eval()
        self.STNru.to(device)
        if os.path.exists(stnetru_checkpoint_path):
            # STN.load_state_dict(torch.load('weights/STN_model_Init.pth', map_location=lambda storage, loc: storage))
            # STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
            checkpoint = torch.load(stnetru_checkpoint_path)
            self.STNru.load_state_dict(checkpoint['net_state_dict'])
            print(f' RU STN weights {stnetru_checkpoint_path} has been loaded')
        else:
            raise ValueError(f'Path {stnetru_checkpoint_path} is not found.')

        # spatial transformation net initialization for RU
        self.STNkz = STNet()
        self.STNkz.eval()
        self.STNkz.to(device)
        if os.path.exists(stnetkz_checkpoint_path):
            # STN.load_state_dict(torch.load('weights/STN_model_Init.pth', map_location=lambda storage, loc: storage))
            # STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
            checkpoint = torch.load(stnetkz_checkpoint_path)
            self.STNkz.load_state_dict(checkpoint['net_state_dict'])
            print(f' KZ STN weights {stnetkz_checkpoint_path} has been loaded')
        else:
            raise ValueError(f'Path {stnetkz_checkpoint_path} is not found.')

        # RU LPRnet initialization

        self.lprnetru = LPRNet(class_num=len(CHARS_ru), dropout_rate=0)
        self.lprnetru.eval()
        self.lprnetru.to(device)
        if os.path.exists(lprnetru_checkpoint_path):
            # lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
            checkpoint = torch.load(lprnetru_checkpoint_path)
            self.lprnetru.load_state_dict(checkpoint['net_state_dict'])
            print(f' RU LPRNet weights {lprnetru_checkpoint_path} has been loaded')
        else:
            raise ValueError(f'Path {lprnetru_checkpoint_path} is not found.')

        # KZ LPRnet initialization

        self.lprnetkz = LPRNet(class_num=len(CHARS_kz), dropout_rate=0)
        self.lprnetkz.eval()
        self.lprnetkz.to(device)
        if os.path.exists(lprnetkz_checkpoint_path):
            # lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
            checkpoint = torch.load(lprnetkz_checkpoint_path)
            self.lprnetkz.load_state_dict(checkpoint['net_state_dict'])
            print(f' KZ LPRNet weights {lprnetkz_checkpoint_path} has been loaded')
        else:
            raise ValueError(f'Path {lprnetkz_checkpoint_path} is not found.')

    def execute(self, frame):
        # TODO follow the convention of the components in this solution this method should accept batch of frames
        img_lp_size = (94, 24)
        img_car_size = (24, 24)
        # detection
        if self.detector_lp == 'retinanet':
            image = retina_preprocess(frame)
            image_group_on_device = [torch.tensor(image, dtype=torch.float).to(self.device)]
            detections_group = self.retinanet(image_group_on_device)
            detections = detections_group[0]
            detections_off_device = {'boxes': detections['boxes'].detach().cpu().numpy(),
                                     'scores': detections['scores'].detach().cpu().numpy(),
                                     'cls': detections['labels'].detach().cpu().numpy(),
                                     }
        elif self.detector_lp == 'mtcnn_lpr':
            # preprocessing is performed in execute_mtcnn_net method
            car_frame, car_bboxes = execute_mtcnn_net("CAR", frame, img_car_size, self.device, cls_num=6, pnet=self.pnet_car, onet=self.onet_car)
            car_type = get_class_name(car_bboxes[:, 5])
            frame, bboxes = execute_mtcnn_net("LPR", frame, img_lp_size, self.device, cls_num=2, pnet=self.pnet_lp, onet=self.onet_lp)
            # mtcnn bboxes are filtered with score 0.8
            detections_off_device = {'boxes': bboxes[:, 0:4],
                                     'scores': bboxes[:, 4],
                                     'cls': bboxes[:, 5],
                                     'car_type_scores': car_bboxes[:, 4],
                                     'car_boxes': car_bboxes[:, 0:4],
                                     'car_type': car_type,
                                     }
        else:
            raise ValueError("No detector model in the pipeline.")

        # TODO reconsider necessity to move off device tensor at this stage
        # TODO consider to pack up all the detection on given image into one batch for lprnet

        # alignment and recognition
        detections_num = detections_off_device['boxes'].shape[0]
        # detections_num = detections_off_device['scores'].shape[0]
        numbers = [] * detections_num
        # selection = np.where(detections_off_device['scores'] > score_threshold)[0]
        for i in range(detections_num):  # selection:
            boxes = detections_off_device['boxes'][i]
            detected_lp = frame[int(boxes[1]):int(boxes[3]) + 1, int(boxes[0]):int(boxes[2]) + 1, :]
            height, width, _ = detected_lp.shape
            if height == 0 or width == 0:
                numbers.insert(i, ['not detected'])
                continue
            if height != img_lp_size[1] or width != img_lp_size[0]:
                detected_lp = cv2.resize(detected_lp, img_lp_size)
            detected_lp = lprnet_preprocess(detected_lp)
            detected_lp = np.expand_dims(detected_lp, axis=0)
            detected_lp_on_device = torch.tensor(detected_lp, dtype=torch.float).to(self.device)
            # aligned = detected_lp_on_device

            if detections_off_device['cls'][i] == 1:
                aligned = self.STNru(detected_lp_on_device)
                recognized_lp = self.lprnetru(aligned)  # torch.Size([batch_size, CHARS length, output length ])
                class_name = 'RU'
                chars = CHARS_ru
                recognized_lp_off_device = recognized_lp.cpu().detach().numpy()
            elif detections_off_device['cls'][i] == 2:
                aligned = self.STNkz(detected_lp_on_device)
                recognized_lp = self.lprnetkz(aligned)  # torch.Size([batch_size, CHARS length, output length ])
                class_name = 'KZ'
                chars = CHARS_kz
                recognized_lp_off_device = recognized_lp.cpu().detach().numpy()  # (batch size, alphabet, 18)
            else:
                raise NotImplemented('Unsupported class has been detected.')
            lp, _ = beam_search_decoder(class_name, recognized_lp_off_device, chars, None)  # list of predicted numbers
            # lp, _ = greedy_decoder(recognized_lp_off_device, CHARS)  # list of predicted numbers
            numbers.insert(i, lp)
        detections_off_device['numbers'] = numbers
        # TODO follow the convention of the components in this solution this method should return batch of frames
        #  and batch of corresponding detections
        return frame, detections_off_device

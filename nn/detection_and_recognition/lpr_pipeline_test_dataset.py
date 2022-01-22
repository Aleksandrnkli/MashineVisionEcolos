from commons.crop_lpr_aux_generator import AuxGeneratorForLPRCrop
import torch
from lpr_pipeline import Pipeline
import numpy as np
import argparse


image_extension = '.jpg'
score_threshold = 0.45
iou_threshold = 0.45


def prepare_generator(dataset_path):
    # create the generators
    train_generator = AuxGeneratorForLPRCrop(
        dataset_path,
        'final_test',
        image_extension=image_extension,
        shuffle_groups=False,
        classes={
            'RU plate number': 1,
            'KZ plate number': 2,
        },
    )

    validation_generator = AuxGeneratorForLPRCrop(
        dataset_path,
        'test',
        image_extension=image_extension,
        shuffle_groups=False,
        classes={
            'RU plate number': 1,
            'KZ plate number': 2,
        },
    )

    return train_generator, validation_generator


def calc_iou(box_true, box_pred):
    # x1 = box[0]; y1 = box[1]; x2 = box[2]; y2 = box[3]
    max_x = max(box_true[0], box_pred[0])
    max_y = max(box_true[1], box_pred[1])
    min_x = min(box_true[2], box_pred[2])
    min_y = min(box_true[3], box_pred[3])

    inter_w = abs(min_x - max_x)
    inter_h = abs(min_y - max_y)
    inter_area = inter_w * inter_h

    w1 = box_true[2] - box_true[0]
    h1 = box_true[3] - box_true[1]
    w2 = box_pred[2] - box_pred[0]
    h2 = box_pred[3] - box_pred[1]
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = float(inter_area)/float(union_area)
    return iou


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Preparing dataset...')
    train_generator, validation_generator = prepare_generator(args.dataset_path)
    print('Preparing networks pipeline...')

    # retina_epoch_to_use = 100  # retina's epoch to use in demo
    num_classes = 2
    # dataset_type = 'pascal_custom'
    #retinanet_checkpoint_path = f'./retinanet/{dataset_type}_weights/retinanet_{retina_epoch_to_use}.pth'
    stnetru_checkpoint_path = args.stnetru_checkpoint_path
    stnetkz_checkpoint_path = args.stnetkz_checkpoint_path
    lprnetru_checkpoint_path = args.lprnetru_checkpoint_path
    lprnetkz_checkpoint_path = args.lprnetkz_checkpoint_path
    onet_checkpoint_path = args.onet_checkpoint_path
    pnet_checkpoint_path = args.pnet_checkpoint_path

    # pipeline = Pipeline(device, num_classes, retinanet_checkpoint_path=retinanet_checkpoint_path, stnet_checkpoint_path=stnet_checkpoint_path, lprnet_checkpoint_path=lprnet_checkpoint_path)
    pipeline = Pipeline(device, num_classes, pnet_checkpoint_path=pnet_checkpoint_path, onet_checkpoint_path=onet_checkpoint_path, stnetru_checkpoint_path=stnetru_checkpoint_path, stnetkz_checkpoint_path=stnetkz_checkpoint_path, lprnetru_checkpoint_path=lprnetru_checkpoint_path, lprnetkz_checkpoint_path=lprnetkz_checkpoint_path)

    iterations = 0
    targets = 0
    matches = 0
    iou_final = 0
    for sample in train_generator:
        iterations += 1
        image_group, annotations_groups = sample
        image, annotations = image_group[0], annotations_groups[0]

        # assert ('boxes' in annotations)
        # assert ('labels' in annotations)
        # assert (annotations['boxes'].shape[0] == annotations['labels'].shape[0])

        _, detections = pipeline.execute(image)

        # assert ('boxes' in detections)
        # assert ('labels' in detections)
        # assert (detections['boxes'].shape[0] == detections['labels'].shape[0])

        print(f'sample # {iterations}')

        selection = np.where(detections['scores'] > score_threshold)[0]
        for i in range(annotations['boxes'].shape[0]):
            targets += 1
            label_true = annotations['platenums'][i][0]
            class_num = annotations['labels'][i]
            box_true = annotations['boxes'][i].astype(int)
            detected = False
            for c, j in enumerate(selection):
                label_pred = detections['numbers'][j][0]
                box_pred = detections['boxes'][j].astype(int)
                iou = calc_iou(box_true, box_pred)
                if label_true == label_pred and iou > iou_threshold:
                    matches += 1
                    detected = True
                    selection = np.delete(selection, c)
                    print(f'HIT: label_true {label_true}, class {class_num}, label_pred {label_pred}; iou {iou}')
                    iou_final += iou
                    break
                elif (label_true != label_pred and iou > iou_threshold) or (label_true == label_pred and iou <= iou_threshold):
                    detected = True
                    print(f'MISS: label_true {label_true}, class {class_num}, label_pred {label_pred}; iou {iou}')
                    iou_final += iou
            if not detected:
                print(f'MISS: label_true {label_true}, not detected')
                iou_final += 0

        if iterations >= train_generator.size():
            # we need to break the loop by hand because
            # the generator loops indefinitely
            print(f'accuracy: {matches/targets}')
            avarage_iou = iou_final/iterations
            print(f'avarage iou: {avarage_iou}')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo pipeline with lpr detection and lpr recognition on the test dataset")
    parser.add_argument("--stnetru_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/RU/stn_Iter_045300_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--stnetkz_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/KZ/stn_Iter_063000_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--lprnetru_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/RU/lprnet_Iter_045300_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--lprnetkz_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/KZ/lprnet_Iter_063000_model.ckpt", help="path to weights for ru_stn")
    parser.add_argument("--onet_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/onet_Weights", help="path to weights for ru_stn")
    parser.add_argument("--pnet_checkpoint_path", default="E:/Models/LPR/LPR_with_mtcnn/pnet_Weights", help="path to weights for ru_stn")
    parser.add_argument("--dataset_path", default="E:/DataSets/LicensePlate/Russian_KZ_LPR_mtcnn_lpr_2cls", help="path to the dataset")
    args = parser.parse_args()
    main(args)

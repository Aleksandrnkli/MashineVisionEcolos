import cv2
import torch
from mtcnn.model.MTCNN import create_mtcnn_net, execute_mtcnn_net
import os
from commons.crop_lpr_aux_generator import AuxGeneratorForLPRCrop
from commons.crop_generator_car_type import AuxGeneratorForCarModeCrop
from settings import get_const_for_MTCNN



def main():
    image_extension = '.jpg'
    dataset_path = 'E:/DataSets/Car_type/Dataset'  # 'C:/Datasets/TEST_LPR'
    dataset_part = 'test'  # 'trainval'

    mode = "CAR"
    _, _, _, mini_lp_size, cls_num, kernel, mp, linear = get_const_for_MTCNN(learning_mode=mode)
    print('Preparing dataset...')
    if mode == "LPR":
        custom_classes = {
            'RU plate number': 1,
            'KZ plate number': 2
        }
        generator = AuxGeneratorForLPRCrop(dataset_path, dataset_part, classes=custom_classes,
                                           image_extension=image_extension, shuffle_groups=True, batch_size=1,
                                           resize_images=False)

    elif mode == "CAR":
        custom_classes = {
            'firetruck': 1,
            'police': 2,
            'ambulance': 3,
            'car': 4,
            'bus': 5,
            'truck': 6,
        }
        generator = AuxGeneratorForCarModeCrop(dataset_path, dataset_part, classes=custom_classes,
                                               image_extension=image_extension, shuffle_groups=False, batch_size=1,
                                               resize_images=False)


    print('Preparing networks pipeline...')
    folder_root = 'D:/LocalProjects/ecosmart/Ecolos/nn/detection_and_recognition/mtcnn/train'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pnet, onet = create_mtcnn_net(linear=linear, kernel=kernel, mp=mp, device=device, cls_num=cls_num, p_model_path=os.path.join(folder_root, 'pnet_Weights'),
                                  o_model_path=os.path.join(folder_root, 'onet_Weights'))

    iterations = 0
    for sample in generator:
        # image_group, annotations_groups = sample
        image_group, _ = sample
        image = image_group[0]
        # annotations = annotations_groups[0]

        # gt_boxes = annotations['boxes']
        # cls_ids = annotations['labels']
        # assert ('boxes' in annotations)
        # assert ('labels' in annotations)
        # assert (annotations['boxes'].shape[0] == annotations['labels'].shape[0] == annotations['platenum'].shape[0])

        # image_paths = generator.image_group_paths(iterations)
        # image_path = image_paths[0]
        # image = cv2.imread(image_path)

        image, annotations_predicted = execute_mtcnn_net(mode, image, mini_lp_size, device, cls_num, pnet, onet)
        for i in range(annotations_predicted.shape[0]):
            bbox = annotations_predicted[i, :4]
            x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
            # w = int(x2 - x1 + 1.0)
            # h = int(y2 - y1 + 1.0)
            # score = annotations_predicted[i, 4]
            cls = annotations_predicted[i, 5]

            if cls == 1:
                color = (255, 0, 0)
                cls_name = 'firetruck'
            elif cls == 2:
                color = (0, 255, 0)
                cls_name = 'police'
            elif cls == 3:
                color = (0, 0, 255)
                cls_name = 'ambulance'
            elif cls == 4:
                color = (0, 255, 255)
                cls_name = 'car'
            elif cls == 5:
                color = (255, 255, 0)
                cls_name = 'bus'
            elif cls == 6:
                color = (255, 0, 255)
                cls_name = 'truck'
            else:
                color = (0, 0, 0)

            cv2.resize(image, (800, 600))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        
        cv2.imshow('image', image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        print(f'dataset instance # {iterations}')
        iterations += 1
        if iterations >= generator.size():
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break


if __name__ == '__main__':
    main()


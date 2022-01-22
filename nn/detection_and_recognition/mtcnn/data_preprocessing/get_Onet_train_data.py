import cv2
import os
import torch
import numpy as np
from mtcnn.utils.util import IoU
from mtcnn.model.MTCNN import create_mtcnn_net, execute_mtcnn_net
from settings import get_const_for_MTCNN
# uncomment corresponding to the processed dataset line
# from annotations_retriever_ccpd import annotation_retriever
from annotations_retriever_gen import annotation_retriever
from mtcnn.utils import assemble

dataset_dest_root = "E:/DataSets/LicensePlate/mtcnn/"
dataset_source = 'E:/DataSets/Car_type/Dataset'
learning_mode = "CAR"
image_size, _, _, mini_lp_size, cls_num, kernel, mp, _ = get_const_for_MTCNN(learning_mode)
def process_dataset(dataset_source,
                    dataset_part,
                    pos_save_dirname,
                    part_save_dirname,
                    neg_save_dirname,
                    onet_postive_filename,
                    onet_part_filename,
                    onet_neg_filename):

    # store labels of positive, negative, part images
    f1 = open(onet_postive_filename, 'w')
    f2 = open(onet_neg_filename, 'w')
    f3 = open(onet_part_filename, 'w')

    annotations = annotation_retriever(learning_mode, dataset_source, dataset_part)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # ignored
    idx = 0

    pnet, _ = create_mtcnn_net(mp, kernel, device, cls_num=cls_num, p_model_path='../train/pnet_Weights', o_model_path=None)

    for im_path, annotations in annotations.items():
        # assert (annotations['boxes'].shape[0] == annotations['labels'].shape[0]) == annotations['platenums'].shape[0])
        if annotations['boxes'].shape[0] == 0:
            continue

        boxes = annotations['boxes']
        cls_ids = annotations['labels']

        print(im_path)

        image = cv2.imread(im_path)


        image, bboxes = execute_mtcnn_net(learning_mode, image, mini_lp_size, device, cls_num, pnet=pnet, onet=None)
        dets = np.round(bboxes[:, 0:4])

        if dets.shape[0] == 0:
            continue

        img = cv2.imread(im_path)
        idx += 1

        img_height, img_width, img_channel = img.shape

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img_width - 1 or y_bottom > img_height - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, boxes)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, image_size, interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3 and n_idx < 3.2*p_idx+1:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dirname, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx_Iou = np.argmax(Iou)
                assigned_gt = boxes[idx_Iou]
                cls_id = cls_ids[idx_Iou]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dirname, "%s.jpg" % p_idx)
                    f1.write(save_file + ' %d %.2f %.2f %.2f %.2f\n' % (cls_id, offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4 and d_idx < 1.2*p_idx + 1:
                    save_file = os.path.join(part_save_dirname, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':

    for dataset_part in ['trainval', 'test']:
        print(f'processing {dataset_part}...')

        if not os.path.exists(f"{dataset_dest_root}/{dataset_part}/"):
            os.mkdir(f"{dataset_dest_root}/{dataset_part}/")
        if not os.path.exists(f"{dataset_dest_root}/{dataset_part}/24/"):
            os.mkdir(f"{dataset_dest_root}/{dataset_part}/24/")

        pos_save_dirname = f"{dataset_dest_root}/{dataset_part}/24/positive"
        part_save_dirname = f"{dataset_dest_root}/{dataset_part}/24/part"
        neg_save_dirname = f"{dataset_dest_root}/{dataset_part}/24/negative"

        if not os.path.exists(pos_save_dirname):
            os.mkdir(pos_save_dirname)
        if not os.path.exists(part_save_dirname):
            os.mkdir(part_save_dirname)
        if not os.path.exists(neg_save_dirname):
            os.mkdir(neg_save_dirname)

        pnet_postive_filename = os.path.join(dataset_dest_root, f'pos_24_{dataset_part}.txt')
        pnet_part_filename = os.path.join(dataset_dest_root, f'part_24_{dataset_part}.txt')
        pnet_neg_filename = os.path.join(dataset_dest_root, f'neg_24_{dataset_part}.txt')

        imglist_filename = f'{dataset_dest_root}/imglist_anno_24_{dataset_part}.txt'

        process_dataset(dataset_source,
                        dataset_part,
                        pos_save_dirname,
                        part_save_dirname,
                        neg_save_dirname,
                        pnet_postive_filename,
                        pnet_part_filename,
                        pnet_neg_filename)

        anno_list_filenames = [pnet_postive_filename, pnet_part_filename, pnet_neg_filename]

        chose_count = assemble.assemble_data(imglist_filename, anno_list_filenames)
        print("PNet train annotation result file path:%s" % imglist_filename)

        print(f'processing of {dataset_part} has been complete.')







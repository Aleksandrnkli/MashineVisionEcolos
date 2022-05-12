import cv2
import os
import numpy as np
from mtcnn.utils.util import IoU
from settings import get_const_for_MTCNN
# uncomment corresponding to the processed dataset line
# from annotations_retriever_ccpd import annotation_retriever
from annotations_retriever_gen import annotation_retriever
from mtcnn.utils import assemble

dataset_dest_root = "E:/DataSets/LicensePlate/mtcnn"
dataset_source = 'E:/DataSets/Car_type/Dataset'
# cls_num = 2 in this part of the logic isn't used explicitly

learning_mode = "CAR"
_, W, H, _, _, _, _, _ = get_const_for_MTCNN(learning_mode)
def process_dataset(dataset_source,
                    dataset_part,
                    pos_save_dirname,
                    part_save_dirname,
                    neg_save_dirname,
                    pnet_postive_filename,
                    pnet_part_filename,
                    pnet_neg_filename):

    # store labels of positive, negative, part images
    f1 = open(pnet_postive_filename, 'w')
    f2 = open(pnet_neg_filename, 'w')
    f3 = open(pnet_part_filename, 'w')

    annotations = annotation_retriever(learning_mode, dataset_source, dataset_part)

    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # ignored
    idx = 0
    for im_path, annotations in annotations.items():
        # assert (annotations['boxes'].shape[0] == annotations['labels'].shape[0]) == annotations['platenums'].shape[0])
        if annotations['boxes'].shape[0] == 0:
            continue

        boxes = annotations['boxes']
        cls_ids = annotations['labels']

        print(im_path)

        img = cv2.imread(im_path)
        idx += 1

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 35:
            size_x = np.random.randint(W, min(width, height) / 2)
            size_y = np.random.randint(H, min(width, height) / 2)
            nx = np.random.randint(0, width - size_x)
            ny = np.random.randint(0, height - size_y)
            crop_box = np.array([nx, ny, nx + size_x, ny + size_y])

            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny: ny + size_y, nx: nx + size_x, :]
            resized_im = cv2.resize(cropped_im, (W, H), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dirname, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box, cls_id in zip(boxes, cls_ids):
            # box (x_left, y_top, w, h)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # generate negative examples that have overlap with gt
            for i in range(5):
                size_x = np.random.randint(W, min(width, height) / 2)
                size_y = np.random.randint(H, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size_x, -x1), w)
                delta_y = np.random.randint(max(-size_y, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size_x > width or ny1 + size_y > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size_x, ny1 + size_y])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size_y, nx1: nx1 + size_x, :]
                resized_im = cv2.resize(cropped_im, (W, H), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dirname, "%s.jpg" % n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
            # generate positive examples and part faces
            for i in range(20):
                size_x = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                size_y = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size_x / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size_y / 2, 0)
                nx2 = nx1 + size_x
                ny2 = ny1 + size_y

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size_x)
                offset_y1 = (y1 - ny1) / float(size_y)
                offset_x2 = (x2 - nx2) / float(size_x)
                offset_y2 = (y2 - ny2) / float(size_y)

                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (W, H), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dirname, "%s.jpg" % p_idx)
                    f1.write(save_file + ' %d %.2f %.2f %.2f %.2f\n' % (cls_id, offset_x1, offset_y1, offset_x2, offset_y2))  # https://pyformat.info/
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4 and d_idx < 1.2*p_idx + 1:
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
        if not os.path.exists(f"{dataset_dest_root}/{dataset_part}/12/"):
            os.mkdir(f"{dataset_dest_root}/{dataset_part}/12/")

        pos_save_dirname = f"{dataset_dest_root}/{dataset_part}/12/positive"
        part_save_dirname = f"{dataset_dest_root}/{dataset_part}/12/part"
        neg_save_dirname = f"{dataset_dest_root}/{dataset_part}/12/negative"

        if not os.path.exists(pos_save_dirname):
            os.mkdir(pos_save_dirname)
        if not os.path.exists(part_save_dirname):
            os.mkdir(part_save_dirname)
        if not os.path.exists(neg_save_dirname):
            os.mkdir(neg_save_dirname)

        pnet_postive_filename = os.path.join(dataset_dest_root, f'pos_12_{dataset_part}.txt')
        pnet_part_filename = os.path.join(dataset_dest_root, f'part_12_{dataset_part}.txt')
        pnet_neg_filename = os.path.join(dataset_dest_root, f'neg_12_{dataset_part}.txt')

        imglist_filename = f'{dataset_dest_root}/imglist_anno_12_{dataset_part}.txt'

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


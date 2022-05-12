import argparse
import torch
from mtcnn.model.MTCNN_nets import PNet, ONet
import math
from mtcnn.utils.util import *
import cv2
import time
from mtcnn.preprocessing import mtcnn_preprocess
from settings import get_const_for_execute_mtcnn_net


def create_mtcnn_net(linear, kernel, mp, device, cls_num, p_model_path=None, o_model_path=None):
    pnet, onet = None, None

    if p_model_path is not None:
        pnet = PNet(mp=mp, kernel=kernel, cls_num=cls_num).to(device)
        pnet.load_state_dict(torch.load(p_model_path))
        pnet.eval()

    if o_model_path is not None:
        onet = ONet(linear=linear, cls_num=cls_num).to(device)
        onet.load_state_dict(torch.load(o_model_path))
        onet.eval()

    return pnet, onet


# TODO consider to refactor encapsulation these method in a class and pass-in 'cls_num' to ctor. For now it doesn't seem right to pass in 'cls_num' in the method
def execute_mtcnn_net(learning_mode, image, mini_lp_size, device, cls_num, pnet=None, onet=None):
    stride, cell_size, size = get_const_for_execute_mtcnn_net(learning_mode)
    predicted_annotations = []
    if pnet:
        image, predicted_annotations = detect_pnet(pnet, image, mini_lp_size, device, cls_num, stride, cell_size)
    if onet and len(predicted_annotations) > 0:
       image, predicted_annotations = detect_onet(onet, image, predicted_annotations, device, cls_num, size)
    return image, predicted_annotations


def detect_pnet(pnet, image, min_lp_size, device, cls_num, stride, cell_size):

    # start = time.time()

    thresholds = 0.6  # lp detection thresholds
    nms_thresholds = 0.7

    # BUILD AN IMAGE PYRAMID
    height, width, channel = image.shape
    min_height, min_width = height, width

    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    factor_count = 0
    while min_height > min_lp_size[1] and min_width > min_lp_size[0]:
        scales.append(factor ** factor_count)
        min_height *= factor
        min_width *= factor
        factor_count += 1

    # it will be returned
    predicted_annotations = []

    with torch.no_grad():
        # run P-Net on different scales
        for scale in scales:
            sw, sh = math.ceil(width * scale), math.ceil(height * scale)
            img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
            # save_file = f'C:/Datasets/1_{scale}.jpg'
            # cv2.imwrite(save_file, img)

            img = mtcnn_preprocess(img)
            img = np.expand_dims(img, 0)

            img = torch.FloatTensor(img).to(device)

            offsets, probs = pnet(img)
            offsets = offsets.cpu().data.numpy()  # offsets: transformations to true bounding boxes
            probs = probs.cpu().data.numpy()
            for cls in range(1, cls_num+1):
                probs_cls = probs[0, cls, :, :]  # probs: probability of a target obj of a given cls at each sliding window
                # applying P-Net is equivalent, in some sense, to moving 12x44 window with strides 2, 5
                stride, cell_size = stride, cell_size
                # indices of boxes where there is probably a lp
                # returns a tuple with an array of row indexes and an array of col indexes:
                inds = np.where(probs_cls > thresholds)

                if inds[0].size == 0:
                    continue

                # transformations of bounding boxes
                tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
                offsets_cls = np.array([tx1, ty1, tx2, ty2])
                score = probs_cls[inds[0], inds[1]]

                # P-Net is applied to scaled images so we need to rescale bounding boxes back
                annotations_at_scale_for_cls = np.vstack([
                    np.round((stride[1] * inds[1] + 1.0) / scale),
                    np.round((stride[0] * inds[0] + 1.0) / scale),
                    np.round((stride[1] * inds[1] + 1.0 + cell_size[1]) / scale),
                    np.round((stride[0] * inds[0] + 1.0 + cell_size[0]) / scale),
                    score,
                    np.ones_like(score)*cls,
                    offsets_cls])
                annotations_at_scale_for_cls = annotations_at_scale_for_cls.T

                predicted_annotations.append(annotations_at_scale_for_cls)

        # collect boxes (and offsets, and scores) from different scales
        # predicted_annotations = [b for b in predicted_annotations if b is not None]

        # NMS
        if predicted_annotations != []:
            predicted_annotations = np.vstack(predicted_annotations)
            # TODO why NMSed non-calibrated bboxes? consider to move here the calibrate_box() invocation
            keep = nms(predicted_annotations[:, 0:6], nms_thresholds)
            predicted_annotations = predicted_annotations[keep]
        else:
            predicted_annotations = np.zeros((1,10))

        # use offsets predicted by pnet to transform bounding boxes
        predicted_annotations = calibrate_box(predicted_annotations[:, 0:6], predicted_annotations[:, 6:])
        # shape [n_boxes, 6]. at each of the 6 positions: x1, y1, x2, y2, score, cls_index

        predicted_annotations[:, 0:4] = np.round(predicted_annotations[:, 0:4])
        predicted_annotations[:, 5] = np.round(predicted_annotations[:, 5])
        # print("pnet predicted in {:2.3f} seconds".format(time.time() - start))
        return image, predicted_annotations


def detect_onet(onet, image, predicted_annotations_1stage, device, cls_num, size):

    # start = time.time()

    thresholds = 0.8  # detection thresholds
    nms_thresholds = 0.7
    height, width, channel = image.shape

    num_boxes = len(predicted_annotations_1stage)
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(predicted_annotations_1stage, width, height)

    img_boxes = np.zeros((num_boxes, 3, size[1], size[0]))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(img_box, size, interpolation=cv2.INTER_LINEAR)

        img_box = mtcnn_preprocess(img_box)
        img_box = np.expand_dims(img_box, 0)

        img_boxes[i, :, :, :] = img_box

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    offsets, probs = onet(img_boxes)
    offsets = offsets.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = probs.cpu().data.numpy()  # shape [n_boxes, cls_num+1]

    keeps = []
    scores_2stage = []
    class_indexes_2stage = []
    for cls in range(1, cls_num + 1):
        keep = np.where(probs[:, cls] > thresholds)[0]
        score_2stage = probs[keep, cls].reshape((-1,))
        class_index_2stage = np.ones_like(score_2stage) * cls
        keeps.append(keep)
        scores_2stage.append(score_2stage)
        class_indexes_2stage.append(class_index_2stage)
    keeps = np.hstack(keeps)
    scores_2stage = np.hstack(scores_2stage)
    class_indexes_2stage = np.hstack(class_indexes_2stage)

    predicted_annotations_2stage = predicted_annotations_1stage[keeps]
    if predicted_annotations_2stage.size != 0:
        predicted_annotations_2stage[:, 4] = scores_2stage  # assign score from stage 2
        predicted_annotations_2stage[:, 5] = class_indexes_2stage  # assign cls from stage 2
        offsets = offsets[keeps]

        predicted_annotations_2stage = calibrate_box(predicted_annotations_2stage, offsets)
        keep = nms(predicted_annotations_2stage, nms_thresholds, mode='min')
        predicted_annotations_2stage = predicted_annotations_2stage[keep]
        predicted_annotations_2stage[:, 0:4] = np.round(predicted_annotations_2stage[:, 0:4])
        predicted_annotations_2stage[:, 5] = np.round(predicted_annotations_2stage[:, 5])
    # print("onet predicted in {:2.3f} seconds".format(time.time() - start))
    return image, predicted_annotations_2stage


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MTCNN Demo')
    parser.add_argument("--test_image", dest='test_image', help="test image path", default="C:/Datasets/000003573.jpg", type=str)
    parser.add_argument("--scale", dest='scale', help="scale the image", default=1, type=int)
    parser.add_argument("--cls_num", dest='cls_num', help="num of classes including background cls", default=2, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help="Minimum lp to be detected. derease to increase accuracy. Increase to increase speed", default=(50, 15), type=int)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(args.test_image)
    image = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)

    start = time.time()

    # pnet, onet = create_mtcnn_net(device, args.cls_num, p_model_path='../train/pnet_Weights', o_model_path='../train/onet_Weights')
    pnet = PNet(cls_num=args.cls_num).to(device)
    onet = ONet(cls_num=args.cls_num).to(device)

    predicted_annotations = execute_mtcnn_net("LPR", image, args.mini_lp, device, args.cls_num, pnet, onet)

    print("image predicted in {:2.3f} seconds".format(time.time() - start))

    for i in range(predicted_annotations.shape[0]):
        bbox = predicted_annotations[i, :4]
        cls = predicted_annotations[i, 5]
        if cls == 1:
            color = (0, 0, 255)
        elif cls == 2:
            color = (255, 0, 0)
        else:
            color = (0, 0, 0)
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        
    image = cv2.resize(image, (0, 0), fx=1/args.scale, fy=1/args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

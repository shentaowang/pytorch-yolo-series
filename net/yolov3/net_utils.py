from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    # predict_transform function takes an detection feature map and turns
    # it into a 2-D tensor which is (batch_size) x (h * w * anchors) x (5+c))
    # suppose the image is square
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # make the bbox attras as a dim, flatten the grid
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors,
                                 grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors,
                                 bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # sigmoid the x, y coordinate and the object score
    # the element in bbox is x, y, score
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # add the center offests
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # change for anchors width and height
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    # change for classification
    prediction[:, :, 5:5+num_classes] = torch.sigmoid(prediction[:, :, 5:5+num_classes])

    # resize to the original image size
    prediction[:, :, :4] *= stride

    return prediction

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes

    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,
                             min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def write_results(prediction, confidence, num_classes, nms = True, nms_conf=0.4):
    # only use the bbox a high level confidence
    # outpt is bboxes * (batch_index, xmin, ymin, xmax, ymax, conf, max_conf, clas)
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        # choose the maximum class
        max_conf, max_conf_index = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_index)
        image_pred = torch.cat(seq, 1)

        non_zero_gred = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_gred, :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:, -1])
        for cls in img_classes:
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            # filter the class is cls
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maxium objectness
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            # number of detections
            idx = image_pred_class.size(0)

            if nms:
                for i in range(idx):
                    # get the ious of all boxes that come after tha one we are
                    # looking at in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0),
                                        image_pred_class[i+1:])
                    except ValueError:
                        break
                    except IndexError:
                        break
                    # zreo out all the detections that have iou > treshold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask

                    #remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0





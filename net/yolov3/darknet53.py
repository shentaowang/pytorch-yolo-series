from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

from .net_utils import predict_transform, write_results, bbox_iou


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class RegionLoss(nn.Module):
    def __init__(self, yolo_block, net_info):
        super(RegionLoss, self).__init__()
        self.num_classes = int(yolo_block['classes'])
        self.num_anchors = int(yolo_block['num'])
        self.ignore_thresh = float(yolo_block['ignore_thresh'])
        self.width = int(net_info['width'])
        self.height = int(net_info['height'])
        self.batch = int(net_info['batch'])
        self.coord_scale = 1
        self.object_scale = 5
        self.noobject_scale = 1
        self.class_scale = 1


    def forward(self, output, target):
        '''
        Preprocess true boxes to training input format
        suppose target shape is
        batch_size x num_anchors x (5)
        which is (xmin, ymin, xmax, ymax, c)

        suppose output shape is batch_size x num_anchors x (5+classes)
        '''
        loss = 0

        map1 = output.size(1)//((1+4+16)*3)
        batch_size = output.size(0)
        # use for loc the grd_grid
        stride = int(self.width//math.sqrt(map1))

        assert map1 * (1+4+16) * 3 == output.size(1)

        basi_anchors = map1 * 3
        map_list = np.array([0, basi_anchors, basi_anchors*4, basi_anchors*16])
        assert stride == 32
        stride_list = [stride, stride//4, stride//16]
        gride_num = int(math.sqrt(map1))
        assert gride_num == 13
        gride_list = [gride_num, gride_num*4, gride_num*16]


        assert map_list.shape == (4, )
        map_list = np.cumsum(map_list, axis=0)
        for l in range(3):
            map = output[:, map_list[l]:map_list[l+1], :]
            # reshpe to batch_size x (h * w) x 3 x (5+classes)
            map = map.view(batch_size, (2**l)*map1, 3, -1)
            # calculate the conf_mask
            conf_mask = torch.ones(map.size()[0:-1])*math.sqrt(self.noobject_scale)
            cls_mask = torch.zeros(map.size()[0:-1])
            coord_mask = torch.zeros(map.size()[0:-1])
            tx = torch.zeros(map.size()[0:-1])
            ty = torch.zeros(map.size()[0:-1])
            tw = torch.zeros(map.size()[0:-1])
            th = torch.zeros(map.size()[0:-1])
            tconf = torch.zeros(map.size()[0:-1])
            tcls = torch.zeros(*map.size()[0:-1], self.num_classes)
            for b in range(batch_size):
                for t in range(target.size(1)):
                    loc_x = (target[b, t, 0] + target[b, t, 2])/2
                    loc_y = (target[b, t, 1] + target[b, t, 3])/2
                    index = (torch.floor(loc_y/stride_list[l]) * gride_list[l]\
                            + torch.floor(loc_x/stride_list[l])).long()
                    output_ = torch.zeros([3, 4])
                    output_[:, 0] = map[b, index, :, 0] - map[b, index, :, 2]/2
                    output_[:, 1] = map[b, index, :, 1] - map[b, index, :, 3]/2
                    output_[:, 2] = map[b, index, :, 0] + map[b, index, :, 2]/2
                    output_[:, 3] = map[b, index, :, 1] + map[b, index, :, 3]/2
                    cur_target = torch.FloatTensor(target[b, t, :4]).repeat(3, 1)
                    ious = bbox_iou(output_, cur_target)
                    conf_mask[b, index][ious > self.ignore_thresh] = 0

                    max_index = torch.argmax(ious)
                    max_iou = torch.max(ious)
                    conf_mask[b, index, max_index] = math.sqrt(self.object_scale)
                    cls_mask[b, index, max_index] = 1
                    coord_mask[b, index, max_index] = 1

                    tconf[b, index, max_index] = max_iou
                    tcls[b, index, max_index, int(target[b, t, 4])] = 1
                    tx[b, index, max_index] = loc_x
                    ty[b, index, max_index] = loc_y
                    tw[b, index, max_index] = target[b, t, 2] - target[b, t, 0]
                    th[b, index, max_index] = target[b, t, 3] - target[b, t, 1]
            print(conf_mask)
            print(map[:, :, :, 4])
            print(tconf)
            print(map[:, :, :, 4]*conf_mask)
            print(tconf*conf_mask)
            loss_conf = nn.MSELoss(size_average=False)(map[:, :, :, 4]*conf_mask, tconf*conf_mask)/2.0
            cls_mask = (cls_mask == 1)
            cls = map[:, :, :, 5:]
            loss_cls = nn.MSELoss(size_average=False)(cls[cls_mask], tcls[cls_mask])/2.0
            loss_cls = self.class_scale * loss_cls

            # rescale coord to 0-1
            map_x = map[:, :, :, 0]/self.width
            map_y = map[:, :, :, 1]/self.width
            map_w = map[:, :, :, 2]/self.width
            map_h = map[:, :, :, 3]/self.width
            tx, ty, tw, th = tx/self.width, ty/self.width, tw/self.width, th/self.width
            loss_x = nn.MSELoss(size_average=False)((2-tw*th)*map_x*coord_mask,
                                                    (2-tw*th)*tx*coord_mask)/2.0
            loss_y = nn.MSELoss(size_average=False)((2-tw*th)*map_y*coord_mask,
                                                    (2-tw*th)*ty*coord_mask)/2.0
            loss_w = nn.MSELoss(size_average=False)((2-tw*th)*map_w*coord_mask,
                                                    (2-tw*th)*tw*coord_mask)/2.0
            loss_h = nn.MSELoss(size_average=False)((2-tw*th)*map_h*coord_mask,
                                                    (2-tw*th)*th*coord_mask)/2.0
            loss_coord = self.coord_scale*(loss_x + loss_y + loss_w + loss_h)
            print(loss_coord)
            print(loss_cls)
            print(loss_conf)
            loss = loss + loss_coord + loss_cls + loss_conf
            return loss




class Darknet53(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet53, self).__init__()
        self.block_list = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.block_list)
        self.loss = RegionLoss(self.block_list[-1], self.net_info)

    def forward(self, x, CUDA):
        modules = self.block_list[1:]
        outputs = {}    #cache the outputs for the route layer
        write = 0
        for i, module in enumerate(modules):
            module_type = (module['type'])

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # get the input dimensions
                inp_dim = int(self.net_info['height'])

                # get the number of classes
                num_classes = int(module['classes'])

                # transform the output data
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.block_list[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.block_list[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

def parse_cfg(cfgfile):
    """
    Take the yolov3 config file

    Return a list of blocks.Each block describe a block in the neural network
    to be bulid.Blocks are represent as a dictionary in the list
    """

    # read the file and get all the useful lines
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.lstrip().rstrip() for x in lines]

    block = {}
    block_list = []

    for x in lines:
        if x[0] == '[':
            if len(block) != 0:
                block_list.append(block)
                block = {}
            block["type"] = x[1:-1].lstrip().rstrip()
        else:
            key, value = x.split('=')
            block[key.rstrip()] = value.lstrip()

    block_list.append(block)

    return block_list


def create_modules(block_list):
    # Captures the information about the input and pre-processing
    net_info = block_list[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(block_list[1:]):
        module = nn.Sequential()
        # check the type of block
        # create a new module for the block
        # append to module_list
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad,
                             bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # If it's an upsampling layer
        # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


def get_test_input():
    img = cv2.imread('../../images/dog.jpg')
    img = cv2.resize(img, (416, 416))
    # BGR -> RGB | H * W C -> C * H * W
    img = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img = img[np.newaxis, :, :, :]/255.0
    img = torch.FloatTensor(img)
    return img

if __name__ == '__main__':
    cfgfile = '../../cfg/yolov3.cfg'
    model = Darknet53(cfgfile)
    model.load_weights('../../bin/yolov3.weights')
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    detect = write_results(pred, confidence=0.5, num_classes=80)
    detect = torch.cat((detect[:, 1:5], detect[:, 7].unsqueeze(1)), 1).unsqueeze(0)
    print(model.loss(pred, detect))


























from __future__ import division
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import os.path as osp
import pandas as pd
import pickle as pkl
from torch.autograd import Variable

from net.yolov3.net_utils import *
from net.yolov3.darknet53 import Darknet53
from defaults import arg_parse
from utils.loader import load_classes
from utils.saver import write_bbox
from utils.transform import letterbox_img, prep_image


args = arg_parse()
dataset = args.dataset
batch_size = int(args.batch_size)
conf_thresh = float(args.conf_thresh)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes('data/coco.names')

# initial the network
print('Loading the network......')
model = Darknet53(args.cfgfile)
model.load_weights(args.weightsfile)
print('Network sucessfult initial')

model.net_info['height'] = args.reso
inp_dim = int(model.net_info['height'])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there is GPU, put the model in GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

read_dir = time.time()
# detection phase
try:
    img_list = [osp.join(osp.realpath('.'), dataset, img) for img
    in os.listdir(dataset)]
except NotADirectoryError:
    img_list = []
    img_list.append(osp.join(osp.realpath('.'), dataset))
except FileNotFoundError:
    print('No file or dictionary with name {}'.format(dataset))
    exit()

if not osp.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_imgs = [cv2.imread(x) for x in img_list]

# pytorch tensor for images
img_batches = list(map(prep_image, loaded_imgs,
                       [inp_dim for x in range(len(img_list))]))

# list contain dimension of original images,as width and height
img_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]
img_dim_list = torch.FloatTensor(img_dim_list).repeat(1, 2)

if CUDA:
    img_dim_list = img_dim_list.cuda()

#create the batchses

leftover = 0
if(len(img_dim_list)) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(img_list) // batch_size + leftover
    img_batches = [torch.cat((img_batches[i*batch_size:
                                          min((i+1)*batch_size, len(img_list))]
                             for i in range(num_batches)))]

write = 0
start_det_loop = time.time()
for i, batch in enumerate(img_batches):
    #load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    with torch.no_grad():
        prediction = model(batch, CUDA)

    prediction = write_results(prediction, conf_thresh, num_classes,
                               nms_conf=nms_thresh)
    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(img_list[i*batch_size:
        min((i + 1)*batch_size, len(img_list))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".
                  format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i*batch_size    #transform the atribute from index in batch to index in imlist

    if not write:                      #If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(img_list[i*batch_size: min((i + 1)*batch_size,
                                                            len(img_list))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".
              format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

try:
    output
except NameError:
    print("No detections were made")
    exit()

img_dim_list = torch.index_select(img_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(inp_dim/img_dim_list, 1)[0].view(-1, 1)


output[:, [1, 3]] -= (inp_dim - scaling_factor*img_dim_list[:, 0].view(-1, 1))/2
output[:, [2, 4]] -= (inp_dim - scaling_factor*img_dim_list[:, 1].view(-1, 1))/2

output[:, 1:5] /= scaling_factor

# make the bbox in image
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim_list[i, 1])

output_recast = time.time()

class_load = time.time()

draw = time.time()
colors = pkl.load(open("data/pallete", "rb"))
list(map(lambda x: write_bbox(x, loaded_imgs, classes, colors), output))


det_names = pd.Series(img_list).apply(lambda x: "{}/det_{}".
                                    format(args.det, x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_imgs))
end = time.time()


print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(img_list)) + " images)",
                               output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img",
                               (end - load_batch)/len(img_list)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()










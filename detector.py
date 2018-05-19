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
import json

from net.yolov3.net_utils import *
from net.yolov3.darknet53 import Darknet53
from defaults import arg_parse
from utils.loader import load_classes
from utils.saver import write_bbox
from utils.transform import prep_image, reverse_resize_img


args = arg_parse()
dataset = args.dataset
batch_size = int(args.batch_size)
conf_thresh = float(args.conf_thresh)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

classes = load_classes('data/coco.names')
num_classes = len(classes)

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
                                          min((i+1)*batch_size, len(img_list))]))
                              for i in range(num_batches)]

write = 0
start_det_loop = time.time()
for i, batch in enumerate(img_batches):
    #load the image and get the ouput
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

    # transform the atribute from index in batch to index in imlist
    prediction[:, 0] += i*batch_size

    # If we have't initialised output
    if not write:
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

output_recast = time.time()

# reverse the resize of image,change output[:, 1:5]
# make the bounding box in the true position in original image
reverse_resize_img(output, inp_dim, img_dim_list)

class_load = time.time()

draw = time.time()
if args.json:
    results_for_json = []
    for i in range(len(img_list)):
        results_for_json.append([])
    for x in output:
        results_for_json[int(x[0])].append(
            {
                "label": classes[int(x[-1])],
                "confidence": float('%.4f' % float(x[5])),
                "topleft": {"x": float(x[1]), "y": float(x[2])},
                "bottomright": {"x": float(x[3]), "y": float(x[4])}
            }
        )
    text_json = [json.dumps(results_for_json[x]) for x in range(len(img_list))]
    det_names = pd.Series(img_list).apply(lambda x: "{}/{}".
                                        format(args.det, x.split("/")[-1].split(".")[0]+'.json'))
    for index, text_file in enumerate(det_names):
        with open(text_file, 'w') as f:
            f.write(text_json[index])
else:
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
print("{:25s}: {:2.3f}".format("Saving Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img",
                               (end - load_batch)/len(img_list)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()










# yolo series
Implement yolo methods, including yolov3

Convert from https://github.com/ayooshkathuria/pytorch-yolo-v3

This repository is trying to achieve the following goals.

- [x] load darknet cfg
- [x] load darknet saved weights
- [x] detect images, output as rectangle in images or json file
- [ ] train the network

## Requirements
Mainly need is:
1. Python 3.5
3. PyTorch 0.4

Others can see in requirements.txt


## Detection Example

![Detection Example](https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.48.42_PM.png)
## Running the detector

### On single or multiple images

Clone, and `cd` into the repo directory. The first thing you need to do is to get the weights file
This time around, for v3, authors has supplied a weightsfile only for COCO [here](https://pjreddie.com/media/files/yolov3.weights), and place

the weights file into your repo directory. Or, you could just type (if you're on Linux)

```
wget https://pjreddie.com/media/files/yolov3.weights
python detector.py --dataset imgs --det det --json 0
```


`--images` flag defines the directory to load images from, or a single image file (it will figure it out), and `--det` is the directory
to save images to. Other setting such as batch size (using `--batch_size` flag) ,object threshold confidence can be tweaked with flags that can be looked up with.

```
python detector.py -h
```

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def arg_parse():
    '''
    parse arguements to the yolo
    '''
    parser = argparse.ArgumentParser(description='yolo detection moduel')
    parser.add_argument('--dataset', dest='dataset', help=
                        'Image / Directory containing images for train test',
                        default='images/', type=str)
    parser.add_argument('--det', dest='det', help='Result to store detections',
                        default='det', type=str)
    parser.add_argument('--json', dest='json', help='Save the detection result as josn file',
                        nargs='?', default=True, type=str2bool)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size',
                        default=1)
    parser.add_argument('--conf_thresh', dest='conf_thresh',
                        help='Objdect confidence to filter', default=0.5)
    parser.add_argument('--nms_thresh', dest='nms_thresh',
                        help='Nms threshold', default=0.4)
    parser.add_argument('--cfg', dest='cfgfile', help='Cfg file',
                        default='cfg/yolov3.cfg', type=str)
    parser.add_argument('--weights', dest='weightsfile', help='weightsfile',
                        default='bin/yolov3.weights', type=str)
    parser.add_argument('--reso', dest='reso', help='Input resolution of iamge',
                        default='416', type=str)

    return parser.parse_args()

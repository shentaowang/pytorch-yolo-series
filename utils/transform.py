import cv2
import torch
import numpy as np

def letterbox_img(img, inp_dim):
    '''
    resize the image with unchage aspect ration using padding
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
    (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    '''
    prepare iamge for inputting to the neural network
    chage BGR to RGB and make 0-255 to 0-1
    return Tensor
    '''
    img = letterbox_img(img, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_

def reverse_resize_img(output, inp_dim, img_dim_list):
    img_dim_list = torch.index_select(img_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / img_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor *
                          img_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor *
                          img_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    # make the bbox in image
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0,
                                        img_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0,
                                        img_dim_list[i, 1])

    return

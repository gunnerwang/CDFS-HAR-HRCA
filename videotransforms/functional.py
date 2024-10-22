import numbers

#import cv2
import numpy as np
import PIL
#from skimage.transform import resize
import torchvision


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
    #    if isinstance(size, numbers.Number):
    #        im_h, im_w = clip[0].shape
    #        # Min spatial dim already matches minimal size
    #        if (im_w <= im_h and im_w == size) or (im_h <= im_w
    #                                               and im_h == size):
    #            return clip
    #        new_h, new_w = get_resize_sizes(im_h, im_w, size)
    #        size = (new_w, new_h)
    #    else:
    #        size = size[1], size[0]
    #    if interpolation == 'bilinear':
    #        np_inter = cv2.INTER_LINEAR
    #    else:
    #        np_inter = cv2.INTER_NEAREST
    #    scaled = [
    #        cv2.resize(img, size, interpolation=np_inter) for img in clip
    #    ]
        raise NotImplementedError
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow

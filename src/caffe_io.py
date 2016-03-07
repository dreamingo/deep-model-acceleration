import numpy as np
import caffe
import cv2
from caffe.proto import caffe_pb2
from google.protobuf import text_format


class IOParam:
    """ The IO parameters for the image preprocessing """
    def __init__(self, over_sample=False, mean_pix=[103.939, 116.779, 123.68],
                 image_dim=256, crop_dim=224, batch_size=10):
        """
        Default parameters are for VGG net
        @Parameters:
            over_sample: bool, whether sample one image mutiple times. if true,
                         the image will be sampled into 10 images(four cropped
                         corners, centers and theirs mirros)
            mean_pix:   The mean pixel of each channel of the imageset;
            image_dim:  resize image, the shorter side is set to image_dim
        """
        self.over_sample = over_sample
        self.mean = mean_pix
        self.image_dim = image_dim
        self.crop_dim = crop_dim
        self.batch_size = 10


def load_net_and_param(net_proto, caffe_model):
    """ Load the `caffe_model` defined in `net_proto`
    @Parameters:
        net_proto: string, filename of the net definition proto file;
        caffe_model: string, filename of the net caffe_model;
    @Returns:
        net: caffe::Net
        net_param: caffe::NetParam
    """
    net = caffe.Net(net_proto, caffe_model, caffe.TEST)
    net_param = load_net_param(net_proto)
    return net, net_param

def construct_net_from_param(net_proto):
    net = caffe.Net(net_proto, caffe.TEST)
    net_param = load_net_param(net_proto)
    return net, net_param

def load_net_param(net_proto):
    """ Construct a NetParameter based on the `net_proto` file """
    net_param = caffe_pb2.NetParameter()
    text_format.Merge(open(net_proto).read(), net_param)
    return net_param


def load_image(img_name):
    """ Load the image with BGR order and with range 0-255 """
    return cv2.imread(img_name)


def transform_image(img, over_sample=False, mean_pix=[103.939, 116.779, 123.68],
                    image_dim=256, crop_dim=224):
    """
    @Parameters:
        img: ndarray with shape (height x width x channel)
        over_sample:
            if over_sample is true, convert one image into 10 images;
            (cropped images of fout corners and the center and the mirrors)
    @Returns:
        Transformed images with shape(#Sample x Channel x Height x Width)
        here sample is 10 if `over_sample` is True else 1;
    """
    # If it's gray-scale, convert it to BGR
    if len(img.shape) < 3 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR)
    # resize image, the shorter side is set to image_dim
    if img.shape[0] < img.shape[1]:
        # Note: OpenCV uses width first...
        dsize = (int(np.floor(float(image_dim) * img.shape[1] / img.shape[0])),
                 image_dim)
    else:
        dsize = (image_dim, int(np.floor(float(image_dim) *
                                         img.shape[0] / img.shape[1])))
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)

    # convert to float32
    img = img.astype(np.float32, copy=False)

    # if over_sample is true, convert one image into 10 images;
    # (cropped images of fout corners and the center and the mirrors)
    if over_sample:
        imgs = np.zeros((10, crop_dim, crop_dim, 3), dtype=np.float32)
    else:
        imgs = np.zeros((1, crop_dim, crop_dim, 3), dtype=np.float32)

    # crop
    indices_y = [0, img.shape[0] - crop_dim]
    indices_x = [0, img.shape[1] - crop_dim]
    center_y = int(np.floor(indices_y[1] / 2))
    center_x = int(np.floor(indices_x[1] / 2))

    imgs[0] = img[center_y:center_y + crop_dim, center_x:center_x + crop_dim, :]
    if over_sample:
        curr = 1
        for i in indices_y:
            for j in indices_x:
                imgs[curr] = img[i:i + crop_dim, j:j + crop_dim, :]
                imgs[curr + 5] = imgs[curr, :, ::-1, :]
                curr += 1
        imgs[5] = imgs[0, :, ::-1, :]

    # subtract mean
    for c in range(3):
        imgs[:, :, :, c] = imgs[:, :, :, c] - mean_pix[c]
    # reorder axis, change shape (num, height, width, channel) to
    # (num, channel, height, width)
    return np.rollaxis(imgs, 3, 1)

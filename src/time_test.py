import argparse
import time
import numpy as np
from caffe_io import load_net_and_param, load_image, transform_image

def parse_args():
    parser = argparse.ArgumentParser(description="Timing the caffe models")
    parser.add_argument('model', type=str, help='protofile of the caffe models')
    parser.add_argument('weights', type=str, help='binary caffe model')
    parser.add_argument('img_list', type=str, help='The image list file')
    parser.add_argument('--batch_num', type=int, default=10,
                        help='Num of images in a batch')
    args = parser.parse_args()
    return  args.model, args.weights, args.img_list, args.batch_num


if __name__ == "__main__":
    model_f, weights_f, img_list_f, batch_num = parse_args()
    # Load the imgs
    print("Loading images...")
    with open(img_list_f, 'r') as f:
        img_list = [ line.strip() for line in f.readlines()]
        imgs = [transform_image(load_image(name)) for name in img_list]

    net, net_param = load_net_and_param(model_f, weights_f)
    imgs_batch = [imgs[i: i+batch_num]for i in xrange(0, len(imgs), batch_num)]
    for ind, batch in enumerate(imgs_batch):
        data = batch[0]
        for img in batch[1::]:
            data = np.vstack((data, img))
        ts = time.time()
        out = net.forward(**{net.inputs[0]: data})
        te = time.time()
        print("Time cost of batch(%d):%2.2f sec" %(ind, te-ts))

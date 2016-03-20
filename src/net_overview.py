# Take an overview of the net, including the 
#     layers data overview: name, shape of blob_data, average, max, min
#     blobs data overview : name, shape of blob_data, average, max, min
from __future__ import print_function
import time
import argparse
import numpy as np
from caffe_io import load_net_and_param, load_image, transform_image
from extrac_conv_layer_response import _extract_response
from common_util import compute_net_complexity, compute_net_storage 


def parse_args():
    parser = argparse.ArgumentParser(description="Overview of the model")
    parser.add_argument('model', type=str, help='protofile of the caffe models')
    parser.add_argument('weights', type=str, help='binary caffe model')
    parser.add_argument('img_list', type=str, help='The image list file')
    parser.add_argument('result_file', type=str, help='The file to write result')
    parser.add_argument('--batch_num', type=int, default=10,
                        help='Num of images in a batch')
    args = parser.parse_args()
    return  (args.model, args.weights, args.img_list, args.result_file,
             args.batch_num)


def blob_overview(data, desc, f):
    print("\t{}({}): max:{}, min:{}, mean:{}".
          format(desc, data.shape, data.max(), data.min(), data.mean()), file=f)


def overview_layers_param(net, net_param, f):
    print("\n------ Overview of the layer parameters --------\n", file=f)
    for ind, layer in enumerate(net.layers):
        name = net_param.layer[ind].name
        print("============ Layer:{} ============".format(name), file=f)
        for i, blob in enumerate(layer.blobs):
            blob_overview(blob.data, "blob{}".format(i), f)


def overview_blobs_param(net, net_param, f):
    print("\n------ Overview of the blobs parameters --------\n", file=f)
    for blob_name in net.blobs:
        print("========== blob response:{} ==========".format(blob_name), file=f)
        blob_overview(net.blobs[blob_name].data, blob_name, f)

def time_forward(net, input_imgbatch, i):
    """ Return the time of foward for a batch """
    ts = time.time()
    print("Forwarding the %d batch..." %(i+1))
    data = input_imgbatch[0]
    for img in input_imgbatch[1::]:
        data = np.vstack((data, img))
    net.forward(**{net.inputs[0]: data})
    te = time.time()
    return te - ts

if __name__ == "__main__":
    model_f, weights_f, img_list_f, result_file, batch_num = parse_args()
    # Load the imgs
    print("Loading images...")
    with open(img_list_f, 'r') as f:
        img_list = [line.strip() for line in f.readlines()]
        imgs = [transform_image(load_image(name)) for name in img_list]

    # Forward a batch to the net
    net, net_param = load_net_and_param(model_f, weights_f)
    imgs_batch = [imgs[i: i+batch_num]for i in xrange(0, len(imgs), batch_num)]
    forward_times = [time_forward(net, imgs_batch[i], i) for i in xrange(0, 10)]

    with open(result_file, 'w') as f:
        print("Overview of NET:{}".format(net_param.name), file=f)
        net_complexity = compute_net_complexity(net, net_param)
        space_complexity = compute_net_storage(net, net_param)
        whole_complexity = sum(net_complexity.itervalues())
        print("Theory complexity:%d" %(whole_complexity), file=f)
        print("Space complexity:%d" %(sum(space_complexity.itervalues())), file=f)
        print("Average time for 10 batch:%.3f" 
              %(sum(forward_times) / len(forward_times)), file=f)
        overview_layers_param(net, net_param, f)
        overview_blobs_param(net, net_param, f)

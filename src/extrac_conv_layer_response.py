# encoding=utf-8
import random
from collections import defaultdict
import numpy as np
from common_util import timeit
from caffe_io import transform_image, IOParam


def extract_conv_response(net, net_param, io_param, imgs, layer_names, 
                          sample_ratio=0.1):
    """ Extract the reponse of the conv layer for the imgs list
    @Parameters:
        net: caffe net used to extract the response
        imgs: list[ndarray], list of image data
        layer_names: list, The layers to extract response from 
        sample_ratio: ratio used to sample column in the responses
                      Since we concatenate all response of the imgs, with
                      result in shape(C, num_imgs * H' * W'). Since it's
                      very large, Therefore we sample a small subset of the
                      columns (`sample_ratio`).
    @Returns:
        reponses: dict, {layers_name:[ responses for each image with
                  shape(C, num_imgs * H' * W' * sample_ratio) ]}
        sample_indices: dict {layers_name:[[sample_indics for batch1,2...]]}
    """
    # Split the image into batches
    batch_size = net.blobs[net.inputs[0]].num
    img_batch = [imgs[i: i + batch_size] for i in xrange(0, len(imgs),
                                                         batch_size)]
    # concatenate responses of mulitple images;
    concat_resp = None
    sample_indices = defaultdict(list)
    for ind, batch in enumerate(img_batch):
        print("Extract resp for %d-%d images..." % (ind * batch_size + 1,
                                                    (ind + 1) * batch_size))
        # responses: {name: (c, batch_num * H' * W' * sample_ratio) ndarray}
        # sample_index: {name: sampel index for this batch(list)}
        responses, sample_index = _extract_response(net, net_param, io_param,
                                                    batch, layer_names, sample_ratio)
        for name in sample_index:
            sample_indices[name].append(sample_index[name])
        if concat_resp is None:
            concat_resp = responses
        else:
            # Concatenate
            for name in responses:
                concat_resp[name] = np.concatenate((concat_resp[name],
                                                    responses[name]), axis=1)
    return concat_resp, sample_indices

@timeit
def _extract_response(net, net_param, io_param, img_batch, layers_name,
                     sample_ratio=0.1, batch_size=10,):
    """ Extract the linear response in certain blobs in the net
    @Parameters:
        img_batch: list, the image list, with len == net.batch_size
        sample_ratio: sample `ratio` column from the response with shape
                      (C, num_batch*H'*W')
        layer_name: list, the name of the layer to the extract;
    @Returns:
        reponses: dict, {blob_name:[ responses for each image with
        shape(C,num_batch*H'*W') ]}
    """
    responses = {}
    indices = {}
    io_param = io_param
    data = transform_image(img_batch[0], io_param.over_sample,
                           io_param.mean, io_param.image_dim,
                           io_param.crop_dim)
    for img in img_batch[1::]:
        data = np.vstack((data, transform_image(img, io_param.over_sample,
                                                io_param.mean,
                                                io_param.image_dim,
                                                io_param.crop_dim)))
    # Do the padding
    if len(img_batch) < batch_size:
        for i in xrange(0, batch_size - len(img_batch)):
            data = np.vstack((data, transform_image(img_batch[0],
                                                    io_param.over_sample,
                                                    io_param.mean,
                                                    io_param.image_dim,
                                                    io_param.crop_dim)))

    for ind, name in enumerate(layers_name):
        start = layers_name[ind - 1] if ind > 0 else None
        try:
            out = net.forward(**{net.inputs[0]: data, 'start': start,
                                       'end': name})
        except KeyError as e:
            top_dict = {l.name: l.top[0] for l in net_param.layer}
            out = {name: net.blobs[top_dict[name]].data}
        # resp with shape(batch_num, c, H',W')
        resp = out[name][0:len(img_batch)]
        # swap axis into (c, batch_num, H',W')
        resp = resp.swapaxes(0, 1)
        column_idx = [i for i in xrange(0, resp.size / resp.shape[0])]
        # Reshape into (c, batch_num * H' * W')
        resp = resp.reshape(resp.shape[0], len(column_idx))
        # Random select `sample_ratio` num columns from `resp`
        random.shuffle(column_idx)
        column_idx.sort()
        column_idx = column_idx[0: int(len(column_idx) * sample_ratio)]
        responses[name] = resp[:, column_idx]
        indices[name] = column_idx
    return responses, indices

# encoding=utf-8
import random
import numpy as np
from common_util import timeit
from caffe_io import transform_image, IOParam


class ConvLayerResponseExtractor:
    """
    A class used to extractor the response of each convolution layer
    in a net
    """
    def __init__(self, net, net_param, io_param=IOParam()):
        self._net = net
        self._net_param = net_param
        self._io_param = io_param
        # 4 is the enum value for the convolution layer type
        conv_layer_params = [l for l in self._net_param.layer if l.type == 'Convolution']
        self._conv_layer_params = conv_layer_params[1::]

    def extract_conv_response(self, imgs, sample_ratio):
        """ Extract the reponse of the conv layer for the imgs list
        @Parameters:
            imgs: list[ndarray], list of image data
            sample_ratio: ratio used to sample column in the responses
                          Since we concatenate all response of the imgs, with
                          result in shape(C, num_imgs * H' * W'). Since it's
                          very large, Therefore we sample a small subset of the
                          columns (`sample_ratio`).
        @Returns:
            reponses: dict, {blob_name:[ responses for each image with
            shape(C, num_imgs * H' * W' * sample_ratio) ]}
        """
        conv_layer_names = [l.name for l in self._conv_layer_params]
        # Split the image into batches
        batch_size = self._net.blobs[self._net.inputs[0]].num
        img_batch = [imgs[i: i + batch_size] for i in xrange(0, len(imgs),
                                                             batch_size)]
        # concatenate responses of mulitple images;
        concat_resp = None
        for ind, batch in enumerate(img_batch):
            print("Extract resp for %d-%d images..." % (ind * batch_size + 1,
                                                        (ind + 1) * batch_size))
            # responses: {name: (c, batch_num * H' * W' * sample_ratio) ndarray}
            responses = self._extract_response(batch, conv_layer_names,
                                               sample_ratio)
            if concat_resp is None:
                concat_resp = responses
            else:
                # Concatenate
                for name in responses:
                    concat_resp[name] = np.concatenate((concat_resp[name],
                                                        responses[name]), axis=1)
        return concat_resp

    @timeit
    def _extract_response(self, img_batch, layers_name, sample_ratio=0.1,
                          batch_size=10,):
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
        io_param = self._io_param
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
            out = self._net.forward(**{self._net.inputs[0]: data, 'start': start,
                                       'end': name})
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
            responses[name] = resp[:, column_idx[0:int(len(column_idx)
                                                       * sample_ratio)]]
        return responses

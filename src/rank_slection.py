from __future__ import print_function
import os
import random
import cPickle
import numpy as np
from sklearn.decomposition import PCA
from caffe_io import transform_image, load_image, IOParam, load_net_and_param
from common_util import get_filters_params, compute_conv_layer_complexity, timeit


class RankSelection:
    def __init__(self, net, net_param, io_param=IOParam()):
        """"""
        self._net = net
        self._net_param = net_param
        self._rank_list = []
        self._io_param = io_param
        # The params {layer_name: common_util.ConvLayerParam} of the filters
        # in each conv layer
        self._filters_params = get_filters_params(net, net_param)
        # 4 is the enum value for the convolution layer type
        conv_layer_params = [l for l in self._net_param.layers if l.type == 4]
        # We do not acclerate conv1_1
        self._conv_layer_params = conv_layer_params[1::]
        self._layer_complexity = {l.name: compute_conv_layer_complexity(l.name,
                                                                        self._filters_params)
                                  for l in self._conv_layer_params}
        self._total_complexity = sum(self._layer_complexity.itervalues())

    def rank_selection(self, imgs, speedup_ratio, response_file):
        """
        Rank selction for the current net for given imgs and speedup-ratio
        @Parameters:
            imgs: list, list of ndarry, image datas;
            speedup_ratio: float
            response_file: The dump filename for the responses, if-exist, load
                           the responses, else, dump it into file after calculation
        @Returns:
            None
        """
        if os.path.isfile(response_file):
            with open(response_file) as f:
                responses = cPickle.load(f)
        else:
            responses = self.extract_conv_response(imgs, 0.1)
            with open(response_file, 'w') as f:
                cPickle.dump(responses, f)
        ranks = self._rank_selection(responses, speedup_ratio)
        with open("./log/result_1000.txt", 'w') as f:
            for name in ranks:
                print("{}:{}".format(name, ranks[name]), file=f)

    @timeit
    def _rank_selection(self, responses, speedup_ratio):
        """
        Select the rank for each conv layer
        @Parameters:
            responses: dict, {blob_name: response(ndarray, (num_filters * (num*H'*W')))}
            speedup_ratio: float, The speed-up ratio for the whole model;
        @Returns:
            ranks: dict, {layer_name: (old_rank, selected_rank)}
        """
        # Compute the eigenvals for each response
        eigenvals = self._blob_eigenvale(responses)
        # {layer_name: eiginevals of the layer response}
        eigenvals = {l.name: eigenvals[l.top[0]] for l in self._conv_layer_params}
        # eigen ratio:
        eigenval_ratio = {name:
                          (eigenvals[name][0:len(eigenvals[name]) / 2].sum() /
                           eigenvals[name].sum()) for name in eigenvals}

        while True:
            # Check if reach to the approximated complexity or not
            appro_complexity = 0.0
            for l in self._conv_layer_params:
                d_ = len(eigenvals[l.name])
                k = self._filters_params[l.name].kernel_size[0]
                d = self._filters_params[l.name].num_filters
                c = self._filters_params[l.name].input_channels
                s_ratio = ((d_ * k * k * c) + (d * d_)) / float(d * k * k * c)
                appro_complexity += s_ratio * self._layer_complexity[l.name]
            if appro_complexity <= self._total_complexity / float(speedup_ratio):
                break
            # Select a eigenvalue and drop it;
            greedy_choice = None
            for l in self._conv_layer_params:
                try:
                    # eigenval relateive reduction
                    obj_reduction = eigenvals[l.name][-1] / eigenvals[l.name].sum()
                    # Complexity reduction
                    comp_reduction = (self._layer_complexity[l.name] /
                                      float(self._filters_params[l.name].num_filters))
                    obj_drop = obj_reduction / comp_reduction
                except IndexError:
                    continue
                if greedy_choice is None or obj_drop < greedy_choice[1]:
                    greedy_choice = (l.name, obj_drop)

            if greedy_choice:
                name = greedy_choice[0]
                eigenvals[name] = eigenvals[name][0:-1]

        ranks = {l: (self._filters_params[l].num_filters, len(eigenvals[l]),
                     eigenval_ratio[l]) for l in eigenvals}
        return ranks

    def extract_conv_response(self, imgs, sample_ratio):
        """ Extract the reponse of the conv layer for the imgs list
        @Parameters:
            imgs: list[ndarray], list of image data
            sample_ratio: ratio used to sample column in the responses
        @Returns:
            reponses: dict, {blob_name:[ responses for each image with
            shape(C,num_imgs*H'*W'*sample_ratio) ]}
        """
        conv_layer_names = [l.name for l in self._conv_layer_params]
        # Split the image into batches
        batch_size = self._net.blobs[self._net.inputs[0]].num
        img_batch = [imgs[i: i + batch_size] for i in xrange(0, len(imgs),
                                                             batch_size)]
        # concatenate responses of mulitple images;
        concat_resp = None
        for ind, batch in enumerate(img_batch):
            print("Extract responses fro %d-%d image..." % (ind * batch_size + 1,
                                                            (ind + 1) * batch_size))
            # responses: {name: (c, batch_num * H' * W' * sample_ratio) ndarray}
            responses = self._extract_response(batch, conv_layer_names, sample_ratio)
            if concat_resp is None:
                concat_resp = responses
            else:
                # Concatenate
                for name in responses:
                    concat_resp[name] = np.concatenate((concat_resp[name],
                                                        responses[name]), axis=1)
        return concat_resp

    def _blob_eigenvale(self, responses):
        """ Compute the eigenvalue of the response in each blob
        @Parameters:
            responses: dict, {blob_name(str), response(ndarray)}
        @Returns:
            eigenvals: dict, {blob_name(str), eigenval(ndarray, sorted)}
        """
        eigenvals = {}
        blobs = self._net.blobs
        for blob in responses:
            size = responses[blob].size
            blob_channels = blobs[blob].channels
            pca = PCA()
            pca.fit(responses[blob].reshape(blob_channels,
                                            size / blob_channels))
            evals = pca.explained_variance_
            eigenvals[blob] = evals
        return eigenvals

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
                                                    io_param.mean, io_param.image_dim,
                                                    io_param.crop_dim)))
        # Do the padding
        if len(img_batch) < batch_size:
            for i in xrange(0, batch_size - len(img_batch)):
                data = np.vstack((data, transform_image(img_batch[0], io_param.over_sample,
                                                        io_param.mean, io_param.image_dim,
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
            responses[name] = resp[:, column_idx[0:int(len(column_idx) * sample_ratio)]]
            # responses[name] = resp
        return responses


if __name__ == "__main__":
    net, net_param = load_net_and_param("../../models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt",
                                        "../../models/vgg16/VGG_ILSVRC_16_layers.caffemodel")
    rs = RankSelection(net, net_param)
    with open('./data/1000_1_per_class.txt') as f:
        img_names = [line.strip() for line in f.readlines()]
    imgs = [load_image(img) for img in img_names]
    rs.rank_selection(imgs, 2.0, "./data/response/VGG16_1000Image_response.pl")

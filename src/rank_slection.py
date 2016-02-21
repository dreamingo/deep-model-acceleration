# enconding=utf-8
from __future__ import print_function
from sklearn.decomposition import PCA
from caffe_io import load_image, load_net_and_param
from common_util import get_filters_params, compute_conv_layer_complexity, timeit
from extrac_conv_layer_response import ConvLayerResponseExtractor


class RankSelection:
    def __init__(self, net, net_param):
        """"""
        self._net = net
        self._net_param = net_param
        self._rank_list = []
        # The params {layer_name: common_util.ConvLayerParam} of the filters
        # in each conv layer
        self._filters_params = get_filters_params(net, net_param)
        # 4 is the enum value for the convolution layer type
        conv_layer_params = [l for l in self._net_param.layer if l.type == 'Convolution']
        # We do not acclerate conv1_1
        self._conv_layer_params = conv_layer_params[1::]
        # Layer complexity for each layer
        self._layer_complexity = {l.name: compute_conv_layer_complexity(l.name,
            self._filters_params) for l in self._conv_layer_params}
        # Total complexity of all conv layers
        self._total_complexity = sum(self._layer_complexity.itervalues())

    @timeit
    def rank_selection(self, eigenvals, speedup_ratio):
        """
        Select the rank for each conv layer
        @Parameters:
            eigenvals: dict, {layer_name: [eigenvals list]}
            speedup_ratio: float, The speed-up ratio for the whole model;
        @Returns:
            ranks: dict, {layer_name: (old_rank, selected_rank, eigenval_ratio)}
        """
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

    @timeit
    def blob_eigenvale(self, responses):
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


if __name__ == "__main__":
    # Load the net and net_param
    net, net_param = load_net_and_param(
            "../../models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt",
            "../../models/vgg16/VGG_ILSVRC_16_layers.caffemodel")
    rank_sel = RankSelection(net, net_param)
    response_extractor = ConvLayerResponseExtractor(net, net_param)

    # Read the image
    with open('./data/input/3000_3_per_class.txt') as f:
        img_names = [line.strip() for line in f.readlines()]
    imgs = [load_image(img) for img in img_names]

    # Calculate the responses
    responses = response_extractor.extract_conv_response(imgs, 0.1)
    # Compute the eigenvals for each response
    eigenvals = rank_sel.blob_eigenvale(responses)

    from collections import defaultdict
    import cPickle
    # {blob_names: {speedup-ratio: [non-3d-decomp-rank, 3d-decomp-rank]}
    batch_ranks = defaultdict(lambda: defaultdict(list))
    speedup_ratio = [2.0, 3.0, 4.0, 8.0]
    for ratio in speedup_ratio:
        print("Rank-selction for ratio:{}".format(ratio))
        rank = rank_sel.rank_selection(eigenvals, ratio)
        rank2 = rank_sel.rank_selection(eigenvals, ratio**0.5)
        for blob_name in rank:
            batch_ranks[blob_name][ratio].append(rank[blob_name][1])
            batch_ranks[blob_name][ratio].append(rank2[blob_name][1])

    with open("./data/rank_selection/rank-sel-3000_VGG16_2-4.pl", 'w') as f:
        cPickle.dump(dict(batch_ranks), f)

    with open("./data/rank_selection/rank-sel-3000_VGG16_2-4.txt", "w") as f:
        for blob_name in batch_ranks:
            print(blob_name, file=f)
            for ratio in batch_ranks[blob_name]:
                print("\t{}:{}".format(ratio, batch_ranks[blob_name][ratio]),
                      file=f)

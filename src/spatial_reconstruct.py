# Implement the Scheme2 decomposition method in Jadergber et al's paper[1]
# Decompose the filters W(d, c, k, k) into two filters set(conv layer)
#     H(d__, c, k, 1)
#     V(d, d__, 1, k)
# [1] Jaderberg, Max, Andrea Vedaldi, and Andrew Zisserman. "Speeding up 
# convolutional neural networks with low rank expansions." 
# arXiv preprint arXiv:1405.3866 (2014).
import cPickle
from collections import defaultdict
import numpy as np
from caffe_io import load_net_and_param
from common_util import matrix_factorization, get_filters_params, timeit

class SaptialDecomposer:
    def __init__(self, net, net_param, rank_file):
        self._net = net
        self._net_param = net_param
        # {layer_name: FilterParams}
        self._filters_params = get_filters_params(net, net_param)
        with open(rank_file) as f:
            # {layer_name: {speedup-ratio:[2d-decomp-rank, 3d-decomp-rank, d__]}}
            self._ranks = cPickle.load(f)
        
    def spatial_decompose_conv_layers(self, dump_file):
        """ Spatial decompose for the convolution layer to be approximated
        """
        decompose_result = defaultdict(lambda: defaultdict())
        speedup_ratios = [2.0, 3.0, 4.0]
        for name in self._ranks:
            for ratio in speedup_ratios:
                print("Decomposing {} with speedup {}x...".format(name, ratio))
                W = self._net.params[name][0].data
                d__ = self._ranks[name][ratio][2]
                H, V = self.spatial_decompose(W, name, d__)
                decompose_result[name][ratio] = (H, V)
                print("Spatil decompose {}({}) into H({}), V({})"
                      .format(name, W.shape, H.shape, V.shape))

        with open(dump_file, 'w') as f:
            cPickle.dump(dict(decompose_result), f)


    @timeit
    def spatial_decompose(self, W, layer_name, d__):
        """ 
        Decompose filters W(d,c,k,k) into two filters
            H(d__, c, k, 1),
            V(d, d__, 1, k)
        @Parameters:
            W: numpy ndarray with shape(d, c, k, k)
            d__, int
        @Returns:
            H: numpy ndarray with shape(d__, c, k, 1)
            V: numpy ndarray with shape(d, d__, 1, k)
        """
        filter_param = self._filters_params[layer_name]
        d = filter_param.num_filters
        c = filter_param.input_channels
        k = filter_param.kernel_size[0]
        # becomes (c,k,k,d)
        W_ = W.swapaxes(0, 2).swapaxes(0, 1).swapaxes(2, 3)
        # becomes ((c*k), (k,d))
        W_ = W_.reshape((c * k, k * d))
        # H in shape((c*k*1),d__), V in shape(d__, (1*k*d))
        H, V = matrix_factorization(W_, feat_num = d__, lr = 0.005, max_iter=300,
                                    err_tol = 0.001)
        # reshape H into (d__, c, k, 1)
        H = H.transpose().reshape((d__, c, k, 1))
        # reshape V into(d, d__, 1, k)
        V = V.reshape((d__ * k, d)).transpose().reshape((d, d__, 1, k))
        return H, V


def vgg16_spatial_reconstruct():
    net, net_param = load_net_and_param(
            "../../models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt",
            "../../models/vgg16/VGG_ILSVRC_16_layers.caffemodel")
    sp = SaptialDecomposer(net, net_param, 
                           "./data/rank_selection/rank-sel-1000_VGG16_2-8.pl")
    sp.spatial_decompose_conv_layers("./data/spatial_decompose/vgg16_spatial_reconstruct.pl")


if __name__ == "__main__":
    vgg16_spatial_reconstruct()

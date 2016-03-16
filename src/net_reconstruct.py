# encoding=utf-8
# 根据rank-selection的结果，生成新的网络结构protofile
from __future__ import print_function
import cPickle
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from caffe_io import load_net_param
from common_util import compute_d__


class NetReconstructor:
    """
    Generate a new-network protofile based on the rank-selection result
    For `3ddecomp` method, the rank-selection support param `d_` and `d__`, then
    a layer is split into three conv layers:
        [(d__, (k, 1), (pad, 0)), (d_, (1, k), (0, pad)), (d, (1, 1),(0, 0))]
        
    For `2ddecomp` method, the rank-selection support param `d_`, then a layer
    is splited into two conv layers:
        [(d_, (k, k)), (d, (1, 1), (0, 0))]

    For `spatial-decomp` method, we calcuate the `d__` based on the speedup_ratio
    then a layer is splited into two conv layers:
        [(d__, (k, 1), (pad, 0)), (d, (1, k), (0, pad))]
    """
    def __init__(self, speedup_ratio, net, net_param, trainval_param, 
                 decomp_flag="3ddecomp"):
        self._speedup_ratio = speedup_ratio
        self._decomp_flag =decomp_flag 
        self._net = net
        self._net_param = net_param
        self._new_net_param = caffe_pb2.NetParameter()
        self._ranks = None
        self._last_output_num = 0
        self._trainval_param = trainval_param

    def net_reconstruct(self, rank_file, deploy_prototxt, trainval_prototxt):
        """
        @Parameter:
            rank_file: str, the filename of the rank-selection pickle result;
            deploy_prototxt: str, the filename  of the new deploy prototxt;
            trainval_prototxt: str, the filename of the new trainval prototxt;
        @Returns:
            None
        """
        self._load_rank(rank_file)
        self._net_reconstruct()
        trainval_net_param = self._generate_trainval_prototxt()
        with open(deploy_prototxt, 'w') as f:
            print(str(self._new_net_param), file=f)

        with open(trainval_prototxt, 'w') as f:
            print(str(trainval_net_param), file=f)

    def _generate_trainval_prototxt(self):
        trainval_net_param = caffe_pb2.NetParameter()
        trainval_net_param.name = "{}_trainval".format(self._new_net_param.name)
        # Construct the data layer
        data_layer = trainval_net_param.layer.add()
        data_layer.CopyFrom(self._trainval_param.layer[0])

        # do not copy the last `prob` layer
        for ind in xrange(len(self._new_net_param.layer) - 1):
            layer_param = trainval_net_param.layer.add()
            layer_param.CopyFrom(self._new_net_param.layer[ind])

        loss_layer = trainval_net_param.layer.add()
        loss_layer.CopyFrom(self._trainval_param.layer[-3])

        top1_ac_layer = trainval_net_param.layer.add()
        top1_ac_layer.CopyFrom(self._trainval_param.layer[-2])

        top5_ac_layer = trainval_net_param.layer.add()
        top5_ac_layer.CopyFrom(self._trainval_param.layer[-1])
        return trainval_net_param

    def _net_reconstruct(self):
        """
        Reconstruct the net, which will split each conv layer into 3 conv layers
        @Returns:
            None
        """
        # Configure the new name and the input
        self._new_net_param.name = "{}_new".format(self._net_param.name)
        self._new_net_param.input.extend(self._net_param.input)
        self._new_net_param.input_dim.extend(self._net_param.input_dim)

        first_conv_l = [l for l in self._net_param.layer 
                           if l.type == 'Convolution'][0]
        last_layer_is_split = False
        self._last_output_num = first_conv_l.convolution_param.num_output
        for ind, layer_param in enumerate(self._net_param.layer):
            # We do not acclearate the first convolution layer
            if (layer_param.type == 'Convolution' and 
                layer_param.name != first_conv_l.name):
                self.decomp_conv(layer_param)
                last_layer_is_split = True
            # If it's not a conv layer, just copy from the original one
            else:
                new_layer_param = self._new_net_param.layer.add()
                new_layer_param.CopyFrom(layer_param)
                if last_layer_is_split:
                    last_layer_is_split = False
                    # clear the bottom name in this layer
                    map(lambda k: new_layer_param.bottom.remove(k), 
                        [a for a in new_layer_param.bottom])
                    new_layer_param.bottom.extend(self._new_net_param.layer[-2].top)
                if layer_param.type == 'InnerProduct':
                    self._last_output_num = layer_param.inner_product_param.num_output

    def decomp_conv(self, layer_param):
        # Get the filter paramters
        if len(layer_param.convolution_param.kernel_size):
            k = layer_param.convolution_param.kernel_size[0]
        else:
            k = layer_param.convolution_param.kernel_w
        d = layer_param.convolution_param.num_output
        try:
            pad = layer_param.convolution_param.pad[0]
        except IndexError:
            pad = 0

        if self._decomp_flag == '3ddecomp':
            d_ = self._ranks[layer_param.name][self._speedup_ratio][1]
            d__ = compute_d__(d, k, d_, self._speedup_ratio)
            new_conv_parm = [(d__, (k, 1), (pad, 0)), (d_, (1, k), (0, pad)),
                             (d, (1, 1),(0, 0))]
            self._decomp_conv(layer_param, new_conv_parm, 3)
            print("Decompose {} into {}".format(layer_param.name, new_conv_parm))
        elif self._decomp_flag == '2ddecomp':
            d_ = self._ranks[layer_param.name][self._speedup_ratio][0]
            new_conv_parm = [(d_, (k, k)), (d, (1, 1), (0, 0))]
            self._decomp_conv(layer_param, new_conv_parm, 2)
            print("Decompose {} into {}".format(layer_param.name, new_conv_parm))
        elif self._decomp_flag == 'spatial_decomp':
            c = self._last_output_num
            d__ = int(d * c * k / (self._speedup_ratio * (d + c)))
            new_conv_parm = [(d__, (k, 1), (pad, 0)), (d, (1, k), (0, pad))]
            self._decomp_conv(layer_param, new_conv_parm, 2)
        else:
            print("Unrecongnize decompose flag:{}".format(self._decomp_flag))

        self._last_output_num = d
        
    def _decomp_conv(self, layer_param, new_conv_parm, decomp_num=3):
        """
        Decompose the the convolution layer into `decomp_num` conv new layer,
        and reconfigure their convolution parameters based on speed up ratio
        @Parameter:
            layer_param: LayerParam, The param of the layer to be decompose
            new_conv_parm: list, [(num_filer, (kernel_height, kernel_width))]
                           the new convolution parameters for each new split 
                           layer
            decomp_num: int, the number this conv layer decomposed into
        @Returns:
            None
        """
        conv_layers = [self._new_net_param.layer.add() 
                      for i in xrange(0, decomp_num)]
        for l in conv_layers:
            l.CopyFrom(layer_param)
        new_blobs = ([[name for name in layer_param.bottom]] +
                     [["{}_split{}".format(layer_param.name, i)] 
                      for i in xrange(1, decomp_num + 1)])

        for ind, l in enumerate(conv_layers):
            l.name = "{}_split{}".format(l.name, ind + 1)
            # Clear the bottom and top field in layer param;
            map(lambda k: l.bottom.remove(k), [a for a in l.bottom])
            map(lambda k: l.top.remove(k), [a for a in l.top])
            l.bottom.extend(new_blobs[ind])
            l.top.extend(new_blobs[ind + 1])
            l.convolution_param.num_output = new_conv_parm[ind][0]
            # Clear the old kernel_size
            map(lambda k: l.convolution_param.kernel_size.remove(k),
                [a for a in l.convolution_param.kernel_size])
            l.convolution_param.kernel_size.extend(new_conv_parm[ind][1])
            # pad
            if len(new_conv_parm[ind]) == 3:
                map(lambda k: l.convolution_param.pad.remove(k),
                    [a for a in l.convolution_param.pad])
                l.convolution_param.pad.extend(new_conv_parm[ind][2])

    def _load_rank(self, rank_file):
        """ Load the rank_file """
        with open(rank_file) as f:
            self._ranks = cPickle.load(f)

def draw_net(net_proto_file, output_img_file, rankdir='TB'):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_proto_file).read(), net)
    print('Drawing net to %s' % output_img_file)
    caffe.draw.draw_net_to_file(net, output_img_file, rankdir)


def vgg16_net_reconstruct():
    net_param = load_net_param(
            "../../models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt")
    trainval_param = load_net_param(
            "./data/proto_file/trainval/VGG_ILSVRC_16_layers_trainval.prototxt")
    ranks_file = "./data/rank_selection/rank-sel-1000_VGG16_2-8-new-comp.pl"

    deploy_prototxt_format = "./data/proto_file/deploy/vgg16_{}x_{}_deploy.{}"
    trainval_prototxt_format = "./data/proto_file/trainval/vgg16_{}x_{}_trainval.{}"

    speedup_ratio = [2.0, 3.0, 4.0]
    for ratio in speedup_ratio:
        reconstror3d = NetReconstructor(ratio, None, net_param, trainval_param,
                                        decomp_flag="3ddecomp")
        # 3ddecomp
        reconstror3d.net_reconstruct(ranks_file, 
                deploy_prototxt_format.format(ratio, "3ddecomp", "prototxt"),
                trainval_prototxt_format.format(ratio, "3ddecomp", "prototxt"))

        # 2ddecomp
        reconstror2d = NetReconstructor(ratio, None, net_param, trainval_param,
                                        decomp_flag="2ddecomp")
        reconstror2d.net_reconstruct(ranks_file, 
                deploy_prototxt_format.format(ratio, "2ddecomp", "prototxt"),
                trainval_prototxt_format.format(ratio, "2ddecomp", "prototxt"))

        # spatial decomp
        reconstror2d = NetReconstructor(ratio, None, net_param, trainval_param,
                                        decomp_flag="spatial_decomp")
        reconstror2d.net_reconstruct(ranks_file, 
                deploy_prototxt_format.format(ratio, "spatial-decomp", "prototxt"),
                trainval_prototxt_format.format(ratio, "spatial-decomp", "prototxt"))


if __name__ == "__main__":
    vgg16_net_reconstruct()

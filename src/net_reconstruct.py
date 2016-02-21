from __future__ import print_function
import cPickle
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from caffe_io import load_net_param


class NetReconstructor:
    def __init__(self, speedup_ratio, net, net_param, decomp3d=True):
        self._speedup_ratio = speedup_ratio
        self._decomp3d = decomp3d
        self._net = net
        self._net_param = net_param
        self._new_net_param = caffe_pb2.NetParameter()
        self._ranks = None

    def net_reconstruct(self, rank_file, new_proto_filename):
        """
        @Parameter:
            new_proto_filename: string, filename for the new net_param to write into
        """
        self._rank = self._load_rank(rank_file)
        self._net_reconstruct()
        with open(new_proto_filename, 'w') as f:
            print(str(self._new_net_param), file=f)

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

        first_conv_name = [l.name for l in self._net_param.layer 
                           if l.type == 'Convolution'][0]
        for ind, layer_param in enumerate(self._net_param.layer):
            # We do not acclearate the first convolution layer
            if (layer_param.type == 'Convolution' and 
                layer_param.name != first_conv_name):
                self.decomp_conv(layer_param)
            # If it's not a conv layer, just copy from the original one
            else:
                new_layer_param = self._new_net_param.layer.add()
                new_layer_param.CopyFrom(layer_param)

    def decomp_conv(self, layer_param):
        def compute_d__(layer_param, k, d_, speedup_ratio):
            """ Compute d__ with Jaderberg's paper"""
            d = layer_param.convolution_param.num_output
            return int((((1 + k**2)/(speedup_ratio**0.5)) - 1) * d * d_ /
                       (k * (d + d_)))

        # Get the filter paramters
        if len(layer_param.convolution_param.kernel_size):
            k = layer_param.convolution_param.kernel_size[0]
        else:
            k = layer_param.convolution_param.kernel_w
        d = layer_param.convolution_param.num_output

        if self._decomp3d:
            d_ = self._ranks[layer_param.name][self._speedup_ratio][1]
            d__ = compute_d__(layer_param, k, d_, self._speedup_ratio)
            new_conv_parm = [(d__, (k, 1)), (d_, (1, k)), (d, (1, 1))]
            self._decomp_conv(layer_param, new_conv_parm, 3)
            print("Decompose {} into {}".format(layer_param.name, new_conv_parm))
        else:
            d_ = self._ranks[layer_param.name][self._speedup_ratio][0]
            new_conv_parm = [(d_, (k, k)), (d, (1, 1))]
            self._decomp_conv(layer_param, new_conv_parm, 2)
            print("Decompose {} into {}".format(layer_param.name, new_conv_parm))
        
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
                      for i in xrange(1, decomp_num)]
                     + [[name for name in layer_param.top]])

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
            l.convolution_param.kernel_w = 0
            l.convolution_param.kernel_h = 0
            l.convolution_param.kernel_size.extend(new_conv_parm[ind][1])

    def _load_rank(self, rank_file):
        """ Load the rank_file """
        with open(rank_file) as f:
            self._ranks = cPickle.load(f)

def draw_net(net_proto_file, output_img_file, rankdir='TB'):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_proto_file).read(), net)
    print('Drawing net to %s' % output_img_file)
    caffe.draw.draw_net_to_file(net, output_img_file, rankdir)


if __name__ == "__main__":
    import caffe.draw
    net_param = load_net_param(
            "../../models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt")
    speedup_ratio = [2.0, 3.0, 4.0]
    ranks_file = "./data/rank_selection/rank-sel-3000_VGG16_2-4.pl"
    proto_file_format = "./data/new_proto_file/vgg16_{}x_{}_deploy.{}"
    for ratio in speedup_ratio:
        reconstror3d = NetReconstructor(ratio, None, net_param)
        # 3ddecomp
        reconstror3d.net_reconstruct(ranks_file, 
                proto_file_format.format(ratio, "3ddecomp", "prototxt"))
        # Draw proto file to image
        draw_net(proto_file_format.format(ratio, "3ddecomp", "prototxt"),
                 proto_file_format.format(ratio, "3ddecomp", "jpeg"))

        # 2ddecomp
        reconstror2d = NetReconstructor(ratio, None, net_param, decomp3d=False)
        reconstror2d.net_reconstruct(ranks_file, 
                proto_file_format.format(ratio, "2ddecomp", "prototxt"))
        # Draw proto file to image
        draw_net(proto_file_format.format(ratio, "2ddecomp", "prototxt"),
                 proto_file_format.format(ratio, "2ddecomp", "jpeg"))

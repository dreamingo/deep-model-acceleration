from __future__ import print_function
from caffe.proto import caffe_pb2
from caffe_io import load_net_param


class NetReconstructor:
    def __init__(self, speedup_ratio, net, net_param):
        self._speedup_ration = speedup_ratio
        self._net = net
        self._net_param = net_param
        self._new_net_param = caffe_pb2.NetParameter()

    def net_reconstruct(self, new_proto_filename):
        """
        Reconstruct the net, which will split each conv layer into 3 conv layers
        @Parameter:
            new_proto_filename: string, filename for the new net_param to write into
        @Returns:
            None
        """
        # Configure the new name and the input
        self._new_net_param.name = "{}_new".format(self._net_param.name)
        self._new_net_param.input.extend(self._net_param.input)
        self._new_net_param.input_dim.extend(self._net_param.input_dim)

        for layer_param in self._net_param.layers:
            # 4 is the enum value of the convolution layer type
            if layer_param.type == 4:
                self.decomp_conv(layer_param)
            # If it's not a conv layer, just copy from the original one
            else:
                new_layer_param = self._new_net_param.layers.add()
                new_layer_param.CopyFrom(layer_param)

        with open(new_proto_filename, 'w') as f:
            print(str(self._new_net_param), file=f)

    def decomp_conv(self, layer_param):
        """
        Decompose the the convolution layer into 3 conv new layer, and
        reconfigure their convolution parameters based on speed up ratio
        @Parameter:
            layer_param: The param of the layer to be decompose
        @Returns:
            None
        """
        conv_layers = [self._new_net_param.layers.add() for i in xrange(0, 3)]
        for l in conv_layers:
            l.CopyFrom(layer_param)
        name = layer_param.name
        new_blobs = [layer_param.bottom, ["{}_split1".format(name)],
                     ["{}_split2".format(name)], layer_param.top]
        for ind, l in enumerate(conv_layers):
            l.name = "{}_split{}".format(l.name, ind + 1)
            # Clear the bottom and top field in layer param;
            map(lambda k: l.bottom.remove(k), [a for a in l.bottom])
            map(lambda k: l.top.remove(k), [a for a in l.top])
            l.bottom.extend(new_blobs[ind])
            l.top.extend(new_blobs[ind + 1])


if __name__ == "__main__":
    net_param = load_net_param("../../models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt")
    reconstror = NetReconstructor(3.0, None, net_param)
    reconstror.net_reconstruct("new_vgg16_deploy.prototxt")

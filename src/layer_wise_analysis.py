# encoding=utf-8
# =============================================================================
# We get the original model `A` and the approximated model `B`. 
# In this script we analysis the error of result that specified layer(s) 
# contribute to when approximated those layer(s);
# For an example, we now analysis the error of result that `conv1_2` contribute
#     1. Replace `conv1_2_split*` in model B with `con1_2` in the original model
#        `A` and generate the new model `C`
#     2. We analysis the the result of model `C` and compare to `B`
# =============================================================================
from __future__ import print_function
import argparse
from collections import OrderedDict
from caffe_io import load_net_and_param, construct_net_from_param
from caffe.proto import caffe_pb2

class LayerWiseAnalysis:
    """"""
    def __init__(self, o_net, o_net_param, n_net, n_net_param):
        self.o_net = o_net
        self.o_net_param = o_net_param
        self.n_net = n_net
        self.n_net_param = n_net_param

        self.o_net_layer_dict = {l.name : (o_net.layers[i], l, i)
                for i, l in enumerate(o_net_param.layer)}
        self.n_net_layer_dict = {l.name : (n_net.layers[i], l, i)
                for i, l in enumerate(n_net_param.layer)}

        self.layer_map = OrderedDict()
        for l in self.o_net_param.layer:
            self.layer_map[l.name] = []
        for l in self.n_net_param.layer:
            name = l.name
            origin_name = name.split("_split")[0]
            self.layer_map[origin_name].append(name)

    def analysis(self, keep_layers, n_prototxt, n_modelfile):
        """
        @Parameters:
            keep_layers: set,  the name of the layer to keep in to orignial model
            n_prototxt:  str,  the filename of the new prototxt
            n_modelfile: str,  the filename of the new caffe model
        @Returns:
            None
        """
        # Validate the layer name in the `keep_layers` set
        for name in keep_layers:
            if name not in self.layer_map:
                raise Exception("No such layer:{} in orignial model".format(name))
        
        self.generate_new_protofile(keep_layers, n_prototxt)
        model_c, model_c_param = construct_net_from_param(n_prototxt)
        self.copy_blobs(keep_layers, model_c, model_c_param)
        model_c.save(n_modelfile)

    def copy_blobs(self, keep_layers, model_c, model_c_param):
        """
        """
        def _copy_blob(source_l, target_l):
            assert(len(source_l.blobs) == len(target_l.blobs))
            for i in xrange(len(source_l.blobs)):
                target_l.blobs[i] = source_l.blobs[i]

        for ind, layer in enumerate(model_c_param.layer):
            name = layer.name
            if name in self.o_net_layer_dict:
                _copy_blob(self.o_net_layer_dict[name][0], model_c.layers[ind])
            elif name in self.n_net_layer_dict:
                _copy_blob(self.n_net_layer_dict[name][0], model_c.layers[ind])
            else:
                raise Exception("No such layer:{} in models".format(name))
        

    def generate_new_protofile(self, keep_layers, n_prototxt):
        """
        """
        new_net_param = caffe_pb2.NetParameter()
        new_net_param.name = "{}_layerwise_analy".format(self.o_net_param.name)
        new_net_param.input.extend(self.n_net_param.input)
        new_net_param.input_dim.extend(self.n_net_param.input_dim)
        last_keep_layer = None
        for name in self.layer_map:
            if name in keep_layers:
                layer_param = self.o_net_layer_dict[name][1]
                n_layer_param = new_net_param.layer.add()
                n_layer_param.CopyFrom(layer_param)
                last_keep_layer = name
            else:
                for n_name in self.layer_map[name]:
                    layer_param = self.n_net_layer_dict[n_name][1]
                    n_layer_param = new_net_param.layer.add()
                    n_layer_param.CopyFrom(layer_param)
                    if last_keep_layer is not None:
                        map(lambda k: n_layer_param.bottom.remove(k), 
                            [a for a in n_layer_param.bottom])
                        n_layer_param.bottom.extend([last_keep_layer])
                        last_keep_layer = None

        with open(n_prototxt, 'w') as f:
            print(str(new_net_param), file=f)


def parse_args():
    parser = argparse.ArgumentParser(description="Layer-wise analysis")
    parser.add_argument('o_proto', type=str, help='protofile of models A')
    parser.add_argument('o_weights', type=str, help='binary model A')
    parser.add_argument('n_proto', type=str, help='protofile of models B')
    parser.add_argument('n_weights', type=str, help='binary model B')
    parser.add_argument('keep_layers', type=str, help='kept layers, sperate with `,`')
    parser.add_argument('new_proto', type=str, help='protofile of new models')
    parser.add_argument('new_weights', type=str, help='new binary model')
    args = parser.parse_args()
    return  (args.o_proto, args.o_weights, args.n_proto, args.n_weights,
             set(args.keep_layers.strip().split(',')), args.new_proto,
             args.new_weights)

if __name__ == "__main__":
    o_prototxt, o_net, n_prototxt, n_net, keep_layers, new_prototxt, new_net = \
            parse_args()
    o_net, o_net_param = load_net_and_param(o_prototxt, o_net)
    n_net, n_net_param = load_net_and_param(n_prototxt, n_net)

    layerwise_analy = LayerWiseAnalysis(o_net, o_net_param, n_net, n_net_param)
    layerwise_analy.analysis(keep_layers, new_prototxt, new_net)

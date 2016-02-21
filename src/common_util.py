import time


class ConvLayerParam:
    def __init__(self):
        self.input_dim = (0, 0)
        self.input_channels = 0
        self.num_filters = 0
        self.kernel_size = (0, 0)
        self.output_dim = (0, 0)


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print '\tFunction(%r) %2.2f sec' % (method.__name__, te - ts)
        return result

    return timed


def get_filters_params(net, net_param):
    """
    Get the filters params(num, channels, width, height) for each conv
    layers in the `net`
    @Returns:
        filter_params: dict, {layer_name(str), filter_param(tuple)}
    """
    filter_params = {}
    blobs = net.blobs
    for l in net_param.layer:
        if l.type == "Convolution":
            conv_param = l.convolution_param
            l_param = ConvLayerParam()
            l_param.num_filters = conv_param.num_output
            l_param.input_channels = blobs[l.bottom[0]].channels
            l_param.kernel_size = (conv_param.kernel_size[0],
                                   conv_param.kernel_size[0])
            l_param.input_dim = (blobs[l.bottom[0]].width,
                                 blobs[l.bottom[0]].height)
            l_param.output_dim = (blobs[l.top[0]].width, blobs[l.top[0]].height)
            filter_params[l.name] = l_param
    return filter_params


def compute_conv_layer_complexity(layer_name, filter_params):
    """ Compute the convoluton layer computation complexity
    complexity = dck^2H'W', here :
        d: the number of fileters;
        k: the spatial size of the filters
        H'W' the height and width of the top blobs
    @Parameters:
        layer_name: the name of the conv layer to be computed;
        filter_params
    @Retunrs:
        complexity, int
    """
    l_param = filter_params[layer_name]
    return (l_param.num_filters * l_param.kernel_size[0] *
            l_param.kernel_size[1] * l_param.input_channels *
            l_param.output_dim[0] * l_param.output_dim[1])

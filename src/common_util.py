import time
import numpy as np
from caffe_io import construct_net_from_param
from scipy.linalg import sqrtm, svd
from numpy.linalg import inv, pinv
from caffe.proto import caffe_pb2


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

def compute_d__(d, k, d_, speedup_ratio):
    """ 
    Compute d__ of Jaderberg's paper scheme2 method
    @Parameters:
        d: the number of filter
        k: the spatial size of the filter
        d_: the approxiated number of filter
        speedup_ratio: float
    """
    return int((((1 + k**2)/(speedup_ratio**0.5)) - 1) * d * d_ /
               (k * (d + d_)))


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


def compute_net_complexity(net, net_param):
    """ 
    Compute the net compute complexity, for convolution layer:
        complexity = ndck^2H'W', here :
            d: the number of fileters;
            k: the spatial size of the filters
            H'W' the height and width of the top blobs
    for full-connected layer:
        commlexity = input_blobs.size() * num_output
    @Parameters:
        layer_name: the name of the conv layer to be computed;
        filter_params
    @Retunrs:
        complexity, int
    """
    net_complexity = {}
    for i, layer in enumerate(net_param.layer):
        name = layer.name
        if layer.type == 'InnerProduct':
            bottom_blob_name = layer.bottom[0]
            input_size = net.blobs[bottom_blob_name].data.size
            num_output = layer.inner_product_param.num_output
            net_complexity[name] = input_size * num_output
        elif layer.type == "Convolution":
            d = layer.convolution_param.num_output
            k = layer.convolution_param.kernel_size[0]
            top_blobs_name = layer.top[0]
            bottom_blobs_name = layer.bottom[0]
            output_size = net.blobs[top_blobs_name].data.size
            c = net.blobs[bottom_blobs_name].data.shape[1]
            net_complexity[name] = output_size * k * k * c
        elif layer.type == "Pooling":
            top_blobs_name = layer.top[0]
            output_size = net.blobs[top_blobs_name].data.size
            k = layer.pooling_param.kernel_size
            net_complexity[name] = output_size * k * k
        else:
            net_complexity[name] = 0
    return net_complexity




def matrix_factorization(M, feat_num, lr=0.0001, max_iter=1000, err_tol=0.001,
                         verbose=True):
    """
    Factorize the matrixm `M(n, b)` into two matrix U(n, feat_num),V(feat_num, b)
    @Parameters:
        M: numpy ndarray with shape(n, b)
        feat_num: int, the feature number of the two decomposed matrices;
        lr: float, learning rate
        max_iter: max_iteraition number;
        verbose: whetehr print the log infomation
    @Returns:
        U, V
    """
    n, b = M.shape
    last_loss = None
    # record the continuous number of loss rise
    loss_rise_num = 0
    # Random inilizate this two matrix into range (-0.5, 0.5)
    U = np.random.rand(n, feat_num) - 0.5
    V = np.random.rand(feat_num, b) - 0.5
    print("Split M({}) into U({}), V({})".format(M.shape, U.shape, V.shape))

    for iter_ in xrange( max_iter):
        lr_ = lr / min(((iter_+1) / 100 + 1), 10)
        lr_ = lr
        for i in xrange(n):
            for j in xrange(b):
                eij = M[i][j] - np.dot(U[i,:], V[:,j])
                # for k in xrange(feat_num):
                U[i,:] += lr_ * (2 * eij * V[:, j])
                V[:,j] += lr_ * (2 * eij * U[i, :])
        # Calculate the err loss
        E = M - np.dot(U, V)
        loss = (E * E).sum() / E.size
        if last_loss is not None and loss > last_loss:
            loss_rise_num += 1
        else:
            loss_rise_num = 0
        last_loss = loss
        if verbose:
            print("Loss:{} in iter {}, loss_rise_num:{}".format(loss, iter_,
                                                                loss_rise_num))
        # if loss rise continuously 5 times, stop it;
        if loss < err_tol or loss_rise_num > 5:
            break
    return U, V


def construct_one_layer_tmp_net(o_layer_param, input_shape, W, b, 
                                tmp_layer_name='tmp_conv',
                                tmp_proto_file='tmp_net.prototxt'):
    """
    Construct a tmp net with a convolution layer
    @Parameters:
        o_layer_param: the old layer parameters:
        input_shape: The input shape of the tmp net
        W: the weights of the filters in the tmp_layer
        b: the bias of the filters in the tmp_layer
    @Returns:
        tmp_net, tmp_net_param
    """
    # Construct the tmp net protofile
    # ========================================
    tmp_net_param = caffe_pb2.NetParameter()
    tmp_net_param.name = u'tmp_net'
    # configure the input
    tmp_net_param.input.append(u'data')
    tmp_net_param.input_dim.extend(input_shape)
    # configure the layer
    l_param = tmp_net_param.layer.add()
    l_param.type = "Convolution"
    l_param.name= tmp_layer_name
    l_param.bottom.append('data')
    l_param.top.append(tmp_layer_name)
    l_param.convolution_param.CopyFrom(o_layer_param.convolution_param)
    l_param.convolution_param.kernel_size.insert(0, 1)
    # save it to file
    with open(tmp_proto_file, 'w') as f:
        f.write(str(tmp_net_param))

    net, net_param = construct_net_from_param(tmp_proto_file)
    # Assige W, b to blobs data
    assert(tuple(net.layers[0].blobs[0].shape) == W.shape)
    assert(tuple(net.layers[0].blobs[1].shape) == b.shape)
    net.layers[0].blobs[0].data[...] = W
    net.layers[0].blobs[1].data[...] = b
    return net, net_param

def gsvd(A, Rk, Rl):
    """
    Generalized singular value decomposition[1] Decompose A = USV', 
    with U'KU = I, V'LV = I, and K = Rk x Rk', L = Rl x Rl'
    We let A* = Rk' x A x Rl = U* x D* x V*', We recorver A: 
    A = (Rk')-1 x A* x (Rl)-1, therefore:
        U = (Rk')-1 * U*
        D = D*
        V = (V*' x (Rl)-1)' =(Rl')-1 x V*
    Since Rk' and Rl may be singular, therefore we compute the Moore-Penrose
    g-inverse.
    @Parameters:
        A: a numpy ndarray with shape(I,J)
        Rk: a numpy ndarray with shape(I, k)
        Rl: a numpy ndarray with shape(J, l)
    @Returns:
        U(I,K), S(1,K), V'(K,J), where K = min(I, J)
    @Reference:
        [1]:https://books.google.com/books?id=hXX6AQAAQBAJ&pg=PA65&lpg=PA65&dq=
        SVD+under+nonidentity+metrics+GSVD&source=bl&ots=6fgU0yaHGy&sig=
        yPnypPBCHfCFskjPYIINvJlvNL0&hl=en&sa=X&ved=0ahUKEwiV9IKgz6HLAhVJ-
        2MKHa5qCl0Q6AEIHjAA#v=onepage&q=SVD%20under%20nonidentity%20metrics%20GSVD&f=false
        http://www.sciencedirect.com/science/article/pii/S0167947307000710
    """
    Rk_t = Rk.transpose()
    Rl_t = Rl.transpose()
    A_ = Rk_t.dot(A).dot(Rl)
    U_, D_, V_h = svd(A_, full_matrices=False)
    U = pinv(Rk_t).dot(U_)
    Vh = V_h.dot(pinv(Rl))

    # Check for errors:
    A_approximate = U.dot(np.diag(D_)).dot(Vh)
    Err = A - A_approximate
    print("Loss of gsvd:{:.3f}".format(np.linalg.norm(Err)))
    return U, D_, Vh

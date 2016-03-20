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


def compute_net_storage(net, net_param):
    """ Compute the net space complexity, for covolution layer:
            complexity: DCK1K2
        For full-connected layer:
            complexity: dC
    """
    space_complexity = {}
    for name in net.params:
        W = net.params[name][0].data
        b = net.params[name][1].data
        space_complexity[name] = W.size + b.size
    return space_complexity


def compute_net_complexity(net, net_param):
    """ 
    Compute the net compute complexity, for convolution layer:
        complexity = ndck^2H'W', here :
            d: the number of fileters;
            k: the spatial size of the filters
            H'W' the height and width of the top blobs
    for full-connected layer:
        commlexity = input_blobs.size() * num_output
    @Retunrs:
        net_complexity, dict, {layer_name: complexity}
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
            k1 = layer.convolution_param.kernel_size[0]
            try:
                k2 = layer.convolution_param.kernel_size[1]
            except:
                k2 = k1
            top_blobs_name = layer.top[0]
            bottom_blobs_name = layer.bottom[0]
            output_size = net.blobs[top_blobs_name].data.size
            c = net.blobs[bottom_blobs_name].data.shape[1]
            net_complexity[name] = output_size * k1 * k2 * c
        elif layer.type == "Pooling":
            top_blobs_name = layer.top[0]
            output_size = net.blobs[top_blobs_name].data.size
            k = layer.pooling_param.kernel_size
            net_complexity[name] = output_size * k * k
        else:
            net_complexity[name] = 0
    return net_complexity


def tensor_decompose(name, W, K):
    """
    We reshape W(C*D1, D2*N) and decompose W = VH, the shape of V, H
        V: (C*D1, K)
        H: (K, (D2*N))
    We conduct SVD on W = UDQ_h, and we got:
        V = U[:,0:K] * D^(0.5)
        H = D^(0.5) * Q_h
    After that, we reshape V (K,C, D1, 1), K(N, K, 1, D2)
    @Paramters:
        W:  ndarray with shape(N, C, D1, D2)
        K:  int
    @Returns:
        H, V
    """
    ts = time.time()
    N, C, D1, D2 = W.shape
    # K = min(C*D1, D2*N)
    # W = np.rollaxis(W, 0, 3).reshape(C*D1, D2*N)
    W = np.swapaxes(W, 0, 1).swapaxes(1, 2).swapaxes(2, 3).reshape(C*D1, D2*N)
    U, D, Q_h = np.linalg.svd(W, full_matrices=False)
    D_sqrt = np.diag(D[0:K])**0.5
    V = U[:,0:K].dot(D_sqrt)
    H = D_sqrt.dot(Q_h[0:K,:])

    Err = W - V.dot(H)
    loss = np.linalg.norm(Err)
    V = V.reshape(C, D1, 1, K)
    H = H.reshape(K, 1, D2, N)
    V = np.rollaxis(V, 3, 0)
    H = np.rollaxis(H, 3, 0)
    te = time.time()
    print("Decompose {} cost {:.3f}, with loss:{}".format(name, te-ts, loss))
    return H, V


def construct_one_layer_tmp_net(convolution_param, input_shape, W, b, 
                                tmp_layer_name='tmp_conv',
                                tmp_proto_file='tmp_net.prototxt'):
    """
    Construct a tmp net with a convolution layer, which convolution_param is 
    specified by parameter `convolution_param`
    @Parameters:
        convolution_param: The convolution param of the tmp conv layer
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
    l_param.convolution_param.CopyFrom(convolution_param)
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

def compute_output_shape(input_shape, pad, stride, kernel_shape):
    output_shape = []
    for i in xrange(len(input_shape)):
        input_dim = input_shape[i]
        output_dim = (input_dim + 2 * pad[i] - kernel_shape[i]) / stride[i] + 1
        output_shape.append(output_dim)
    return tuple(output_shape)
        

def img2col(X, pad, stride, kernel_shape, bias):
    """
    Convert the input X(C, H, W) into ((CK^2), H'W')
    """
    H_, W_, = compute_output_shape(X.shape[1::], pad, stride, kernel_shape)
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_shape
    C, H, W = X.shape
    X_ = np.zeros((C, H+pad_h*2, W+pad_w*2))
    end_h = -pad_h if pad_h != 0 else X.shape[1]
    end_w = -pad_w if pad_w != 0 else X.shape[2]
    X_[:, pad_h: end_h, pad_w: end_w] = X
    resultX = []
    for i in xrange(H_):
        start_h = i * stride_h
        end_h = start_h + kernel_h
        for j in xrange(W_):
            start_w = j * stride_w
            end_w = start_w + kernel_w
            tmp_result = X_[:, start_h:end_h, start_w:end_w].flatten()
            if bias is None:
                resultX.append(tmp_result)
            else:
                resultX.append(np.append(tmp_result, 1.0))
    resultX = np.array(resultX).transpose()
    return resultX
   

@timeit
def conduct_convolution(W, b, X, pad=(0,0), stride=(1,1)):
    """
    Conduction The convolution operation on input X
    Y_n = (D ,(CK^2)) * ((CK^2), H'W') = (D, H'W') = (D, H', W')
    @Parameters:
        W: ndarray with shape(D, C, K, K), the filter sets of the convolution
        b: ndarray with shape(1, D)
        X: ndarray with shape(N, C, H, W), the input data

    """
    kernel_shape = W.shape[2::]
    # Reshape W into shape(D, CK^2)
    W = W.reshape(W.shape[0], W.size / W.shape[0])
    # Reshape W into shape(D, CK^2+1)
    if b is not None:
        W = np.concatenate((W, b.reshape(b.size, 1)), axis=1)
    H_, W_ = compute_output_shape(X.shape[2::], pad, stride, kernel_shape)
    Y = []
    for n in xrange(X.shape[0]):
        # X_ with shape(ck^2+1, H'W')
        Xn = img2col(X[n], pad, stride, kernel_shape, b)

        # Yn with shape(D, H'W')
        Yn = W.dot(Xn)
        Yn = Yn.reshape(Yn.shape[0], H_, W_)
        Y.append(Yn)
    return np.array(Y)


if __name__ == "__main__":
    W = np.random.rand(64, 64, 3, 3) - 0.5
    b = np.random.rand(1,64)
    X = np.random.rand(10, 64, 128, 128)
    # X_ = img2col(X, (1, 1), stride=(1,1), kernel_shape=(3,3))
    Y = conduct_convolution(W, b, X, pad=(0,0), stride=(1,1))
    print Y.shape

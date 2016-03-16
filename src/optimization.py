# encoding=utf-8
import logging
import logging.config
import cPickle
from collections import OrderedDict
import numpy as np
from numpy.linalg import inv
from caffe.proto import caffe_pb2
from sklearn.decomposition.pca import PCA
from common_util import (construct_one_layer_tmp_net, timeit, gsvd,
                         tensor_decompose)
from extrac_conv_layer_response import extract_conv_response, _extract_response
from caffe_io import (load_net_and_param, construct_net_from_param, load_image,
                      IOParam)

logger = logging.getLogger('optimizer')
logging.config.fileConfig('./data/config/optimizer_logging_config.ini')

class NetDecompOptimizer:
    """"""
    def __init__(self, o_net, o_net_param, n_net, n_net_param, speedup_ratio,
                 io_param=IOParam()):
        """
        @Parameters:
            o_net: The old net to be approximated
            o_net_param: The net param of the old net
            n_net: The new approximated net;
            n_net_param: The param of the new net;
        """
        self.o_net = o_net
        self.o_net_param = o_net_param
        self.n_net = n_net
        self.n_net_param = n_net_param
        self.speedup_ratio = speedup_ratio
        self.io_param = io_param
        self.last_approximated_layer = None

        self.o_net_layer_dict = {l.name : (o_net.layers[i], l, i)
                for i, l in enumerate(o_net_param.layer)}
        self.n_net_layer_dict = {l.name : (n_net.layers[i], l, i)
                for i, l in enumerate(n_net_param.layer)}
        # {conv1_1: conv1_1, 
        #  conv1_2: [conv1_2_split1, conv1_2_split2, conv1_2_split3]}
        self.layer_map = OrderedDict()
        for l in self.o_net_param.layer:
            self.layer_map[l.name] = []
        for l in self.n_net_param.layer:
            name = l.name
            origin_name = name.split("_split")[0]
            self.layer_map[origin_name].append(name)

    def optimize3d(self, imgs, responses, indices, new_model_file):
        """
        Approximate a convolution layer with three conv layers based on 
        3d-decomposed method on [1]
        @Parameters:
            imgs:      list, The list of image data used for retriving responses
                       to approximate the layer
            responses: dict, {layer_name: responses}, The responses of the imgs
                       of the layers to be approximated from the old net;
            indices:   dict, {layer_name, [indices-for-batch1, ..for-batch2]}
                       The indices that the `responses` sample
            new_model_file: str, the path to save the new caffe model

        @Reference:
            Zhang, Xiangyu, et al. "Accelerating very deep convolutional 
            networks for classification and detection." (2015).
        """
        # Since layer_map is an orderdict, it iter from bottom to top
        for name in self.layer_map:
            logger.debug("\nOptimize layer:{}...".format(name))
            if len(self.layer_map[name]) == 1:
                self._copy_blob(name, self.layer_map[name][0])
            else:
                assert(len(self.layer_map[name]) == 3)
                first_l_name = self.layer_map[name][0]
                W = self.o_net.params[name][0].data
                first_l = self.n_net_layer_dict[first_l_name][1]
                d__ = first_l.convolution_param.num_output
                H, V = tensor_decompose(name, W, d__)
                self._optimize3d_first_l(name, V)
                self._optimize3d_lasttwo_l(name, imgs, responses, indices, H)

        self.n_net.save(new_model_file)

    def optimize2d(self, imgs, responses, sample_indices, new_model_file):
        """
        Approximate a convolution layer with two-conv layers based on 
        2d-decomposed method on [1]
        @Parameters:
            imgs:      list, The list of image data used for retriving responses
                       to approximate the layer
            responses: dict, {layer_name: responses}, The responses of the imgs
                       of the layers to be approximated from the old net;
            indices:   dict, {layer_name, [indices-for-batch1, ..for-batch2]}
                       The indices that the `responses` sample
            new_model_file: str, the path to save the new caffe model

        @Reference:
            Zhang, Xiangyu, et al. "Accelerating very deep convolutional 
            networks for classification and detection." (2015).
        """
        for name in self.layer_map:
            logger.debug("\nOptimize layer:{}...".format(name))
            if len(self.layer_map[name]) == 1:
                self._copy_blob(name, self.layer_map[name][0])
                self.last_approximated_layer = self.layer_map[name][0]
            else:
                assert(len(self.layer_map[name]) == 2)
                self._handle_2d_decomp(name, responses, sample_indices, imgs)
        self.n_net.save(new_model_file)

    def optimize_jaderberg(self, new_model_file):
        """
        """
        def assign_Wb(l_name, W, b=None):
            layer_blobs = self.n_net.params[l_name]
            logger.debug("W.shape:{}".format(W.shape))
            logger.debug("blob[0].shape:{}".format(layer_blobs[0].data.shape))
            print("W.shape:{}, blobs.shape:{}", W.shape, layer_blobs[0].data.shape)
            layer_blobs[0].data[...] = W.reshape(layer_blobs[0].data.shape)
            if b is not None:
                layer_blobs[1].data[...] = b

        for name in self.layer_map:
            logger.debug("\nOptimize layer:{}...".format(name))
            if len(self.layer_map[name]) == 1:
                self._copy_blob(name, self.layer_map[name][0])
                self.last_approximated_layer = self.layer_map[name][0]
            else:
                assert(len(self.layer_map[name]) == 2)
                first_l_name = self.layer_map[name][0]
                K = self.n_net_layer_dict[first_l_name][1].convolution_param.num_output
                # H, V = jaderberg_result[name][self.speedup_ratio]
                W = self.o_net.params[name][0].data
                H, V = tensor_decompose(name, W, K)
                assign_Wb(self.layer_map[name][0], V)
                b = self.o_net.params[name][1].data
                assign_Wb(self.layer_map[name][1], H, b)

        self.n_net.save(new_model_file)


    def _handle_2d_decomp(self, name, responses, sample_indices, imgs,
                          asymmetric=True):
        """
        Decompose the a convolution layer with `name` into 2 conv layers
        (d, c, k, k) -> Q_h(d', c, k, k) and P(d, d', 1, 1)
        @Parameters:
            name:       str,  The name of the layer to be approximated;
            responses:  dict, {name: response}, The responses from old net
            indices:    dict, {layer_name, [indices-for-batch1, ..for-batch2]}
            imgs:       list, The list of image data used for retriving responses
            asymmetric: bool, Whether use asymmetric reconstruction base on [1]
        @Returns:
            None
        """
        first_l_name = self.layer_map[name][0]
        d_ = self.n_net_layer_dict[first_l_name][1].convolution_param.num_output
        Y = responses[name]
        logger.debug("Y.max:{:.2f}, Y.min:{:.2f}".format(Y.max(), Y.min()))
        # # M:(d*d), P:(d*d_), Q_h:(d_*d), b(d*1)
        if asymmetric == True:
            # Construct a one conv-layer tmp net with W and b
            convolution_param = self.o_net_layer_dict[name][1].convolution_param
            input_blob_name = self.o_net_layer_dict[name][1].bottom[0]
            input_shape = self.o_net.blobs[input_blob_name].data.shape
            W = self.o_net.params[name][0].data
            b = self.o_net.params[name][1].data
            tmp_net, tmp_net_param = construct_one_layer_tmp_net(convolution_param, 
                                                                 input_shape, W, b)

            # Get the approximated response Y_ = WX_ + b, and X_ is the response
            # of the layer with name `self.last_approximated_layer`
            Y_ = self.get_approximated_response(tmp_net, 
                                                self.last_approximated_layer,
                                                imgs, sample_indices[name])
        else:
            Y_ = Y

        M, P, Q_h, b = self.init_from_linear_case(Y, d_)
        M, P, Q_h, b = self.alternate_solve(M, b, Y, Y_, d_)
        self._handle_2d_decomp_1l(name, Q_h)
        self._handle_2d_decomp_2l(name, P, b)

    def _handle_2d_decomp_1l(self, name, Q_h):
        W = self.o_net.params[name][0].data
        W = W.reshape(W.shape[0], W.size / W.shape[0])
        b = self.o_net.params[name][1].data
        W = np.concatenate((W, b.reshape(b.size, 1)), axis=1)
        W_ = Q_h.dot(W)
        b_ = W_[:, -1]
        W_ = W_[:,0:-1]
        target_l_name = self.layer_map[name][0]
        target_params = self.n_net.params[target_l_name]
        target_params[0].data[...] = W_.reshape(target_params[0].data.shape)
        target_params[1].data[...] = b_
        self.last_approximated_layer = self.layer_map[name][0]

    def _handle_2d_decomp_2l(self, name, P, b):
        l2_name = self.layer_map[name][1]
        layer_blobs = self.n_net.params[l2_name]
        layer_blobs[0].data[...] = P.reshape(layer_blobs[0].data.shape)
        layer_blobs[1].data[...] = b
        self.last_approximated_layer = self.layer_map[name][1]

    def _copy_blob(self, o_layer_name, n_layer_name):
        """ Copy the blobs data in layer from `o_layer` to `n_layer`"""
        logger.debug("Copying blob data from layer {} to {}"
                     .format(o_layer_name, n_layer_name))
        old_l = self.o_net_layer_dict[o_layer_name][0]
        new_l = self.n_net_layer_dict[n_layer_name][0]
        for ind in xrange(len(old_l.blobs)):
            new_l.blobs[ind] = old_l.blobs[ind]

    def _optimize3d_first_l(self, name, V):
        """
        A conv layer(d, c, k, k) is decompose into 3 layers
            l1: (d__, c, k, 1)
            l2: (d_, d__, 1, k)
            l2: (d, d_, 1, 1)
        @Parameters:
            V: ndarray with shape (d__, c, k, 1)
        The blob data in l1 is optimize by Jaderberg's method
        """
        first_layer_name = self.layer_map[name][0]
        logger.debug("Copy Jarderge decompose result into {}"
                     .format(first_layer_name))
        first_layer = self.n_net_layer_dict[first_layer_name][0]
        assert(V.shape == first_layer.blobs[0].data.shape)
        first_layer.blobs[0].data[...] = V

    def _optimize3d_lasttwo_l(self, name, imgs, responses, sample_indices, H):
        """ Jaderberg's method split a conv layer(d, c, k, k) into 2 layers:
            l1(V) :  (d__, c, k, 1)
            l2_(H): (d, d__, 1, k)
           Here we used [1] method to split `l2_(d, d__, 1, k)` into 2 layers:
            l2: (d_, d__, 1, k)
            l3: (d, d_, 1, 1)
        @Parameters:
            name: The name of the layer to be approximated
            imgs: Image data used to retrieve response
            responses:      The responses from the `old_net` for specified layers
                            {"blob_name": responses(d, num_images*H'*W'*sample_ratio)}
            sample_indices: The sample indices when extract response from the oldnet
                            {"blob_name": [[indics for batch1], [..bathc2], [.]]}
            H:              ndarray with shape (d, d__, 1, k)
        """
        # ===============================================================
        # Construct a tmp net which has a layer W_(d, d__, k, 1), The blobs 
        # data of this net is construct from V in jaderberg_result and 
        # b from original_layer

        tmp_b = self.o_net.params[name][1].data
        input_shape = self.n_net.blobs[self.layer_map[name][0]].data.shape
        l2 = self.n_net_layer_dict[self.layer_map[name][1]][1]
        pad = l2.convolution_param.pad
        conv_param = caffe_pb2.ConvolutionParameter()
        conv_param.kernel_size.extend(H.shape[2::])
        conv_param.pad.extend(pad)
        conv_param.num_output = H.shape[0]
        tmp_net, tmp_net_param = construct_one_layer_tmp_net(conv_param, 
                                                             input_shape, H, tmp_b)
        # Get the approximated response Y_ = WX_ + b, and X_ is the response
        # of the layer with name `self.laye_map[name][0]`
        Y_ = self.get_approximated_response(tmp_net, 
                                            self.layer_map[name][0],
                                            imgs, sample_indices[name])
        d_ = l2.convolution_param.num_output
        Y = responses[name]
        M, P, Q_h, b = self.init_from_linear_case(Y, d_)
        M, P, Q_h, b = self.alternate_solve(M, b, Y, Y_, d_)
        self._handle_3d_decomp_2l(name, Q_h, H, tmp_b)
        self._handle_3d_decomp_3l(name, P, b)
        

    def _handle_3d_decomp_2l(self, name, Q_h, H, tmp_b):
        """
        l2: (d_, d__, 1, k) = Q_h * H
        @Parameters:
            Q_h: ndarray with shape(d_, d)
            H:   ndarray with shape(d, d__, 1, k) -> (d, (d__*1*k))
            tmp_b: ndarray with shape (d, 1)
        """
        l2_name = self.layer_map[name][1]
        # H(d, (d__*1*k))
        H = H.reshape(H.shape[0], H.size / H.shape[0])
        # H(d, (d__*1*k)+1)
        H = np.concatenate((H, tmp_b.reshape(tmp_b.shape[0], 1)), axis=1)
        # W(d_, (d__*1*k)+1)
        W_ = Q_h.dot(H)
        W = W_[:, :-1]
        b = W_[:,-1]
        layer_blobs = self.n_net.params[l2_name]
        layer_blobs[0].data[...] = W.reshape(layer_blobs[0].data.shape)
        layer_blobs[1].data[...] = b

    def _handle_3d_decomp_3l(self, name, P, b):
        """
        l3: (d, d_, 1, 1)
        @Parameters:
            P: d * d_ ndaddary
            b: d * 1 ndarray
        """
        l3_name = self.layer_map[name][2]
        layer_blobs = self.n_net.params[l3_name]
        layer_blobs[0].data[...] = P.reshape(layer_blobs[0].data.shape)
        layer_blobs[1].data[...] = b

    def alternate_solve(self, M, b, Y, Y_, d_):
        """
        Solve Equation argmin_{M, b} || r(Y) - r(MY_ + b) ||2_2
        Which relax the equation into 
            argmin_{M, b,Z} ||r(Y)-r(Z)||2_2 + \lambda ||Z-(MY_+b)||2_2
        We solve M, b and Z alternatively
        @Parameters:
            M:  numpy ndarray with shape(d_ * d_)
            b:  numpy ndarray with shape(d * 1)
            Y:  numpy ndarray with shpe(d * (num_img * H' * W' * sample_ratio))
            Y_: numpy ndarray with shpe(d * (num_img * H' * W' * sample_ratio))
            d_: number of filters of the approximated layer
        @Returns:
            M, P, Q_h, b
        """
        def calculate_nonlinear_loss(Y, Y_, M, b):
            """
            Calculate the nonlinear-loss of the function:
            min || r(Y) - r(MY_ + b) ||_2_2
            """
            rY = np.copy(Y)
            Y__ = M.dot(Y_) + b.reshape((b.shape[0], 1))
            Y__[Y__ < 0] = 0
            Err = rY - Y__
            logger.debug("Loss of Nonlinear-Func:{:.2f}".format(np.linalg.norm(Err)))

        lambda_ = 0.01
        calculate_nonlinear_loss(Y, Y_, M, b)
        for iter_ in xrange(0, 2):
            # if iter_ >= 5: lambda_ = 1.0
            logger.debug("\n==== Alternatively sovling with itr:{}, lambda_:{} ===="
                         .format(iter_, lambda_))
            Z = self.solve_Z(M, b, Y, Y_, lambda_)
            M, P, Q_h, b = self.solve_Mb(M, Z, Y_, d_)
            calculate_nonlinear_loss(Y, Y_, M, b)
        return M, P, Q_h, b

    @timeit
    def solve_Z(self, M, b, Y, Y_, lambda_):
        """
        @Parameters:
            M:  numpy ndarray with shape(d * d)
            b:  numpy ndarray with shape(d * 1)
            Y:  numpy ndarray with shpe(d * (num_img * H' * W' * sample_ratio))
            Y_: numpy ndarray with shpe(d * (num_img * H' * W' * sample_ratio))
        """
        Y__ = M.dot(Y_) + b.reshape((b.shape[0], 1))
        logger.debug("Y__.max:{:.2f}, Y__.min:{:.2f}".format(Y__.max(), Y__.min()))
        Z0 = np.copy(Y__)
        Z0[Z0 > 0] = 0
        # rY = ReLU(Y) rY = max(Y, 0)
        rY = np.copy(Y)
        rY[rY < 0] = 0
        Z1 = (lambda_ * Y__ + rY) / (lambda_ + 1)
        Z1[Z1 < 0] = 0
        Z = np.copy(Z1)
        R0 = rY**2 + lambda_ * (Z0 - Y__)**2
        R1 = (rY-Z1)**2 + lambda_ * (Z1 - Y__)**2
        Z[R1>R0] = Z0[R1>R0]
        logger.debug("Z.max:{:.2f}, Z.min:{:.2f}".format(Z.max(), Z.min()))
        return Z

    @timeit
    def solve_Mb(self, M, Z, Y_, d_):
        """
        @Parameters:
            Z:  numpy ndarray with shpe(d * (num_img * H' * W' * sample_ratio))
            Y_: numpy ndarray with shpe(d * (num_img * H' * W' * sample_ratio))
        @Returns:
            M: d * d ndarray
            P: d * d_ ndaddary, which P.dot(Q_h) = M
            Q_h: d_ * d ndarray, which P.dot(Q_h) = M
            b: d * 1 ndarray
        """
        from scipy.linalg import sqrtm, pinv
        mean_Y_ = Y_.mean(axis=1)
        mean_Z = Z.mean(axis=1)
        Z_ = Z - mean_Z.reshape(mean_Z.shape[0], 1)
        Y__ = Y_ - mean_Y_.reshape(mean_Y_.shape[0], 1)
        Y__cov = Y__.dot(Y__.transpose())
        M_ = Z_.dot(Y__.transpose()).dot(pinv(Y__cov))
        U,S,V = gsvd(M_, np.eye(Y__cov.shape[0]), Y__)
        U_d = U[:, 0:d_]
        S_d = np.diag(S[0:d_])
        V_d = V[0:d_, :]

        M = U_d.dot(S_d).dot(V_d)
        b = mean_Z - np.dot(M, mean_Y_)
        P = U_d.dot(sqrtm(S_d))
        Q_h = sqrtm(S_d).dot(V_d)
        logger.debug("M.max:{:.2f}, M.min:{:.2f}, b.max:{:.2f}, b.min:{:.2f}"
                     .format(M.max(), M.min(), b.max(), b.min()))
        Err = Z_ - M.dot(Y__)
        loss = np.linalg.norm(Err)
        logger.debug("Average loss of |Z-MY|:{:.3f}".format(loss))
        return M, P, Q_h, b

    @timeit
    def init_from_linear_case(self, Y, d_):
        """ Solve the equation min ||(Y-\hat{Y}) - M(Y-\hat{Y})||2_2
        Here we take PCA on Y, which compute the eigen-decomposition on 
            YY^{T} = USU^{T}
        and M = U_{d_} * U_{d_}^{T}, where U_{d_} are the first d_ eignvectors
        and b = \hat{y} - M\hat{y}
        @Parameters:
            Y: ndarray with shape (d, num_imags * H' * W' * sample_ratio)
            d_: the number of eigenvectors to remain
        @Returns:
            M: d * d_
            b = d * 1
        """
        logger.debug("Init M, b from linear-case...")
        pca = PCA(n_components=d_)
        # pca = PCA()
        # with shape d_, * d
        pca.fit(Y.transpose())
        # d_ * d
        U = pca.components_
        # d * d
        M = U.transpose().dot(U)
        mean_Y = np.average(Y, axis=1)
        mean_Y = mean_Y.reshape(mean_Y.shape[0], 1)
        b = mean_Y - M.dot(mean_Y)
        Err = (Y - mean_Y) - M.dot(Y - mean_Y)
        logger.debug("Linear-case loss:{:.3f}".format(np.linalg.norm(Err)))
        logger.debug("Linear-case: M.max:{:.2f}, M.min:{:.2f}, b.max:{:.2f},"
                     " b.min:{:.2f}".format(M.max(), M.min(), b.max(), b.min()))
        return M, U.transpose(), U, b

    def get_approximated_response(self, tmp_net, last_approximated_layer, imgs,
                                  sample_indices):
        """
        Here we compute the Y_ = WX_, here 
            W: The weights of the layer to be approximated
            X_(num_img, c, H, W) is the approximated input to the layer, which
            is also the responses of the params `last_approximated_layer`
        1. We should get X_, it's the response from the layer `last_approximated_layer`
        2. We fit the X_ into the `tmp_net`, and get the response Y_
        Here, we sample `Y_` with the same indices as we sample in `Y`

        @Parameters:
            tmp_net: Caffe::Net, a one-conv layer temp net
            last_approximated_layer: str, the name of the last approximate layer
            imgs: [img_batch1, img_batch2...]
            sample_index: [index-for-b1, index-for-b2..]
        @Returns:
            Y_: the responses of the approximated layer
                with shape(d, num_img * H' * W' * sample_ratio)
        """
        Y_ = None
        batch_size = self.n_net.blobs[self.n_net.inputs[0]].num
        img_batches = [imgs[i: i + batch_size] for i in xrange(0, len(imgs),
                                                               batch_size)]
        for ind, img_batch in enumerate(img_batches):
            top_blobs_name = self.n_net_layer_dict[last_approximated_layer][1].top[0]
            resp_shape = self.n_net.blobs[top_blobs_name].data.shape
            x_, tmp_ind = _extract_response(self.n_net, self.n_net_param, 
                                            self.io_param, img_batch,
                                            [last_approximated_layer],
                                            sample_ratio=1.0)
            # x_ with shape (num_filters, batch_size*H'*Z'*sample_ratio)
            x_ = x_[last_approximated_layer]
            # x_ with shape (num_filters, batch_size, H', Z')
            x_ = x_.reshape((x_.shape[0], len(img_batch), resp_shape[2],
                             resp_shape[3]))
            # x_ with shape (batch_size, num_filters, H', Z')
            x_ = np.swapaxes(x_, 0, 1)
            logger.debug("x_.max:{}, x_.min:{}".format(x_.max(), x_.min()))
            # Feed x_ to tmp_net, to get y_
            out = tmp_net.forward_all(data=x_)
            # y = (c, batch_num, H', W')
            y_ = out['tmp_conv'].swapaxes(0, 1)
            y_ = y_.reshape(y_.shape[0], y_.size / y_.shape[0])
            # sample y_ with the same indices that sample in y
            column_idx = sample_indices[ind]
            y_ = y_[:, column_idx]
            if Y_ is None:
                Y_ = y_
            else:
                Y_ = np.concatenate((Y_, y_), axis=1)
        return Y_


if __name__ == "__main__":
    o_net, o_net_param = load_net_and_param(
            "../../models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt",
            "../../models/vgg16/VGG_ILSVRC_16_layers.caffemodel")
    n_net, n_net_param = construct_net_from_param(
            # "./data/proto_file/deploy/vgg16_4.0x_3ddecomp_deploy.prototxt")
            # "./data/proto_file/deploy/vgg16_4.0x_2ddecomp_deploy.prototxt")
            "./data/proto_file/deploy/vgg16_3.0x_spatial-decomp_deploy.prototxt")

    # with open('./data/input/1000_1_per_class.txt') as f:
    #     img_names = [line.strip() for line in f.readlines()]
    # imgs = [load_image(img) for img in img_names[0:30]]
    #
    # conv_layer_names = [l.name for l in o_net_param.layer 
    #                     if l.type == "Convolution"][1::]
    # responses, sample_indices = extract_conv_response(o_net, o_net_param, IOParam(), 
    #                                                  imgs, conv_layer_names, 
    #                                                  sample_ratio=0.1)


    optimizer = NetDecompOptimizer(o_net, o_net_param, n_net, n_net_param, 4.0)
    # optimizer.optimize3d(imgs, responses, sample_indices,
    #                      "./data/spatial_decompose/vgg16_spatial_reconstruct_new.pl",
    #                      "./data/models/vgg16_4x_3d.caffemodel")
    # optimizer.optimize2d(imgs, responses, sample_indices,
    #                      "./data/models/vgg16_4x_2d.caffemodel")
    optimizer.optimize_jaderberg("./data/models/vgg16_3x_2d_jaderberg.caffemodel")

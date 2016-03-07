from common_util import compute_net_complexity
from caffe_io import load_net_and_param

net, net_param = load_net_and_param(
    "../../../models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt",
    "../../../models/vgg16/VGG_ILSVRC_16_layers.caffemodel")

complexity = compute_net_complexity(net, net_param)
total_complexity = sum(complexity.itervalues())

for l in complexity:
    print "{}:{:3f}".format(l, float(complexity[l]) / total_complexity)

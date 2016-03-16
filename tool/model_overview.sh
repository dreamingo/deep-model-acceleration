CAFFE_ROOT=/Users/dreamingo/work/graduation_project/caffe
PROJECT_ROOT=${CAFFE_ROOT}/play/rank_selection
TIME_TEST_BIN=${PROJECT_ROOT}/src/net_overview.py
IMG_LIST=${PROJECT_ROOT}/data/input/1000_1_per_class.txt

# weights=${CAFFE_ROOT}/models/vgg16/VGG_ILSVRC_16_layers.caffemodel
# prototxt=${CAFFE_ROOT}/models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt
# python ${TIME_TEST_BIN} ${prototxt} ${weights} ${IMG_LIST} ${PROJECT_ROOT}/log/vgg16_overview.txt
#
# weights=${PROJECT_ROOT}/data/models/vgg16_4x_3d.caffemodel
# prototxt=${PROJECT_ROOT}/data/new_proto_file/vgg16_4.0x_3ddecomp_deploy.prototxt
# python ${TIME_TEST_BIN} ${prototxt} ${weights} ${IMG_LIST} ${PROJECT_ROOT}/log/vgg16_4x_3d_overview.txt


# weights=${PROJECT_ROOT}/data/models/vgg16_4x_2d_asym_1k_10iter.caffemodel
# prototxt=${PROJECT_ROOT}/data/new_proto_file/vgg16_4.0x_2ddecomp_deploy.prototxt
# python ${TIME_TEST_BIN} ${prototxt} ${weights} ${IMG_LIST} ${PROJECT_ROOT}/log/vgg16_4x_2d_overview.txt
# #

weights=${PROJECT_ROOT}/data/models/vgg16_3x_2d_jaderberg.caffemodel
prototxt=${PROJECT_ROOT}/data/new_proto_file/vgg16_3.0x_spatial-decomp_deploy.prototxt
python ${TIME_TEST_BIN} ${prototxt} ${weights} ${IMG_LIST} ${PROJECT_ROOT}/log/vgg16_3x_2d_jarderberg_overview.txt

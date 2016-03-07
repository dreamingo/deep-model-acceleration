CAFFE_ROOT=/Users/dreamingo/work/graduation_project/caffe
PROJECT_ROOT=${CAFFE_ROOT}/play/rank_selection
TIME_TEST_BIN=${PROJECT_ROOT}/src/time_test.py
IMG_LIST=${PROJECT_ROOT}/data/input/1000_1_per_class.txt

# weights=${CAFFE_ROOT}/models/vgg16/VGG_ILSVRC_16_layers.caffemodel
# prototxt=${CAFFE_ROOT}/models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt
#
# python ${TIME_TEST_BIN} ${prototxt} ${weights} ${IMG_LIST}

weights=${PROJECT_ROOT}/data/models/vgg16_4x_3d.caffemodel
prototxt=${PROJECT_ROOT}/data/new_proto_file/vgg16_4.0x_3ddecomp_deploy.prototxt
python ${TIME_TEST_BIN} ${prototxt} ${weights} ${IMG_LIST}

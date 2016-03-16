CAFFE_ROOT=/Users/dreamingo/work/graduation_project/caffe
PROJECT_ROOT=${CAFFE_ROOT}/play/rank_selection
ANALYSIS_BIN=${PROJECT_ROOT}/src/layer_wise_analysis.py

ORIGINAL_PROTO=${CAFFE_ROOT}/models/vgg16/VGG_ILSVRC_16_layers_deploy_upgrade.prototxt
ORIGINAL_MODEL=${CAFFE_ROOT}/models/vgg16/VGG_ILSVRC_16_layers.caffemodel

NEW_PROTO=${PROJECT_ROOT}/data/new_proto_file/vgg16_4.0x_2ddecomp_deploy.prototxt
NEW_MODEL=${PROJECT_ROOT}/data/models/vgg16_4x_2d_asym_1k_10iter.caffemodel

TARGET_PROTO=${PROJECT_ROOT}/data/new_proto_file/vgg16_4.0x_2ddecomp_deploy_ana_conv1_2.prototxt
TARGET_MODEL=${PROJECT_ROOT}/data/models/vgg16_4x_2d_asym_1k_10iter_ana_conv1_2.caffemodel

python ${ANALYSIS_BIN} ${ORIGINAL_PROTO} ${ORIGINAL_MODEL} ${NEW_PROTO} \
       ${NEW_MODEL} conv1_2 ${TARGET_PROTO} ${TARGET_MODEL}

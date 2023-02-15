#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=49800 \
tools/test.py configs/SG5/daodixian/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_daodixian.py \
/data/wulianjun/code/CBNetV2/work_dirs/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_daodixian/epoch_9.pth \
--launcher pytorch --eval mAP

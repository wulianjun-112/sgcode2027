#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/SG5/daodixian/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_daodixian_4conv1f_GIoULoss.py

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47810 \
tools/train.py configs/SG5/daodixian/faster_rcnn_r50_fpn_1x_SG5_daodixian_MS_val.py --launcher pytorch 


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47801 \
tools/train.py configs/SG5/daodixian/faster_rcnn_r50_fpn_1x_SG5_daodixian_cluster.py --launcher pytorch 

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47801 \
tools/train.py configs/SG5/daodixian/cascade_rcnn_r50_fpn_1x_SG5_daodixian.py --launcher pytorch 

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47810 \
tools/train.py configs/SG5/daodixian/cascade_rcnn_r2_50_fpn_1x_SG5_daodixian.py --launcher pytorch


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47810 \
tools/train.py configs/SG5/daodixian/faster_rcnn_r50_fpn_1x_SG5_daodixian.py --launcher pytorch 

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47810 \
tools/train.py configs/SG5/daodixian/cascade_rcnn_r101_fpn_1x_SG5_daodixian.py --launcher pytorch 

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47810 \
tools/train.py configs/SG5/daodixian/faster_rcnn_r50_fpn_1x_SG5_daodixian_double_head.py --launcher pytorch  


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=47810 \
tools/train.py configs/SG5/daodixian/cascade_rcnn_r2_101_mdconv_fpn_1x_SG5_daodixian.py --launcher pytorch

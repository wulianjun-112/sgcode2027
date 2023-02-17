#!/bin/bash
export LANG=C.UTF-8
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
if [ ! -x /usr/output_dir/ ]; then
mkdir /usr/output_dir/
fi
python -u tools/test.py --model_types $MODEL_TYPES --output_dir /usr/output_dir/ --show-score-thr $SCORE --eval mAP > /usr/output_dir/test.log
# docker rm -f  `docker ps -aqf "ancestor=sgcode:20230216"`
/bin/bash

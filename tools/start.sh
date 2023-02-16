#!/bin/bash
export LANG=C.UTF-8
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
mkdir /usr/output_dir/
python -u tools/test_docker.py > /usr/output_dir/test.log
/bin/bash

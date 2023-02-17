ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV MODEL_TYPES="00,01"
ENV NMS="0.6"
ENV SCORE="0.3"
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/wulianjun-112/sgcode2027.git /sgcode
WORKDIR /sgcode

# Install MMCV
RUN pip install torch==1.7.0+cu101 torchvision==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
# Install MMDetection
RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
COPY /work_dirs/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_daodixian/epoch_9.pth /sgcode/work_dirs/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_1x_fp16_SG5_daodixian/
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN chmod 777 tools/start.sh
RUN apt update
RUN apt install vim -y
# RUN mkdir -p /usr/local/sgcode/output_dir
ENTRYPOINT ["tools/start.sh"]
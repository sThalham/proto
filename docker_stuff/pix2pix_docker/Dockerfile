# cuda
FROM nvidia/cuda:9.1-runtime-ubuntu16.04
FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda9.1 && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# docker-keras - Keras in Docker with Python 3 and TensorFlow on GPU
FROM gw000/debian-cuda:9.1_7.0
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2018@ena.one>

# install debian packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    # install python 3
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    python3-wheel \
    pkg-config \
    # requirements for numpy
    libopenblas-base \
    python3-numpy \
    python3-scipy \
    # requirements for keras
    python3-h5py \
    python3-yaml \
    python3-pydot \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# manually update numpy
RUN pip3 --no-cache-dir install -U numpy==1.13.3

ARG TENSORFLOW_VERSION=1.5.0
ARG TENSORFLOW_DEVICE=gpu
ARG TENSORFLOW_APPEND=_gpu
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow${TENSORFLOW_APPEND}-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

ARG KERAS_VERSION=2.1.4
ENV KERAS_BACKEND=tensorflow
RUN pip3 --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

# quick test and dump package lists
RUN python3 -c "import tensorflow; print(tensorflow.__version__)" \
 && dpkg-query -l > /dpkg-query-l.txt \
&& pip3 freeze > /pip3-freeze.txt

WORKDIR /srv/



# Clone the Detectron repository
#RUN git clone https://github.com/sThalham/modDetectronCaffe.git /detectron

# Install Python dependencies
#RUN pip install -r /detectron/requirements.txt

# Install the COCO API
#RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
#WORKDIR /cocoapi/PythonAPI
#RUN make install

# Go to Detectron root
#WORKDIR /detectron

# Set up Python modules
#RUN make
#RUN make ops

#CMD ["tools/train_net.py", "--cfg configs/myConfigs/linemodTest_X_101_B64.yaml OUTPUT_DIR detectron/datasets/data/coco/out"]

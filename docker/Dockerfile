########################################################################
### Machine learning Docker image                                    ###
### Ubuntu 16.04, CUDA 9.0, cuDNN v7 and Python3 with following      ###
### Keras, Tensorflow 1.4.1, MXNet                                   ###
########################################################################

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

### 1. Install essential Ubuntu packages, Oracle Java 8, and upgrade pip ###
RUN apt-get update &&\
    apt-get install -y software-properties-common \
    build-essential \
    git \
    wget \
    vim \
    curl \
    zip \
    zlib1g-dev \
    unzip \ 
    pkg-config \
    libblas-dev \
    libopenblas-dev \
    libopencv-dev \
    liblapack-dev \
    python-dev \
    python-pip \
    python-tk \
    python-wheel \
    swig \
    graphviz \
    && pip install --upgrade pip \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# RUN add-apt-repository -y ppa:webupd8team/java && apt-get update &&\
#     echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections &&\
#     echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections &&\
#     apt-get install -y oracle-java8-installer &&\
#     pip install --upgrade pip \
#     && apt-get clean \
#     && apt-get autoremove \
#     && rm -rf /var/cache/apt/archives/* \
#     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /src

### 2. Install Bazel from source ###
ARG BAZEL_VER=0.5.4
ENV BAZEL_VER $BAZEL_VER
ENV BAZEL_INSTALLER bazel-$BAZEL_VER-installer-linux-x86_64.sh
ENV BAZEL_URL https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VER/$BAZEL_INSTALLER
RUN wget $BAZEL_URL &&\
    wget $BAZEL_URL.sha256 &&\
    sha256sum -c $BAZEL_INSTALLER.sha256 &&\
    chmod +x $BAZEL_INSTALLER &&\
    ./$BAZEL_INSTALLER &&\
    rm /src/*

### 3. Install essential Python packages ###
RUN pip --no-cache-dir install \
    numpy \
    matplotlib \
    scipy \
    pandas \
    jupyter \
    scikit-learn \
    scikit-image \
    seaborn \
    graphviz \
    sqlalchemy \
    beautifulsoup4 \
    pillow \
    twisted \
    scapy \
    nose \
    sympy \
    nltk \
    Cython \
    lmdb \
    XlsxWriter \
    h5py \
    python-dateutil \
    joblib \
    lxml \
    xlrd \
    plotly \
    line-profiler

### 4. Get TensorFlow ###
ARG TF_TAG=v1.4.1
ENV TF_TAG $TF_TAG
# Add profiling library support
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
RUN git clone https://github.com/tensorflow/tensorflow --branch ${TF_TAG} --single-branch

### 5. Set env variables for Tensorflow configure script ###
ENV PYTHON_BIN_PATH="/usr/bin/python" TF_NEED_GCP=0 TF_NEED_CUDA=1 TF_CUDA_VERSION=9.0 TF_CUDNN_VERSION=7 TF_CUDA_COMPUTE_CAPABILITIES=6.1 TF_NEED_HDFS=0 USE_DEFAULT_PYTHON_LIB_PATH=1 TF_NEED_OPENCL=0 
ENV GCC_HOST_COMPILER_PATH="/usr/bin/gcc" CUDA_TOOLKIT_PATH="/usr/local/cuda" CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu" CC_OPT_FLAGS="-march=native" TF_NEED_JEMALLOC=1 TF_ENABLE_XLA=0

### 6. Compile Tensorflow ###
RUN cd tensorflow \
    && tensorflow/tools/ci_build/builds/configured GPU \
    && bazel build -c opt --config=monolithic --config=cuda tensorflow/tools/pip_package:build_pip_package \
    && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip \
    && pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl \
    && rm -rf /pip_pkg \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache \
    && cd / && rm -rf tensorflow

### 7. Configure Tensorboard
RUN apt-get update && apt-get install -y supervisor \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY tensorboard.conf /etc/supervisor/conf.d/

### 8. Install Keras and tflearn
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@2.1.3 \
    tflearn==0.3.2 \
    && rm -rf /pip_pkg \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache

### 9. Install MXNet

RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet --branch 0.11.0
RUN cd mxnet \
    && make -j 8 USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 \
    && cd python \
    && python setup.py install \
    && cd / && rm -rf mxnet

### 10. Install PyTorch

RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl \
    && pip install torchvision 
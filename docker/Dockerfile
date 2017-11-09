FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG NPROCS

ENV NUM_CORES ${NPROCS}
ENV INSTALL_ROOT=/opt

RUN apt-get update && apt-get upgrade -y \
&& apt-get install -y ca-certificates curl \
&& apt-get install -y apt-utils libldap2-dev \
&& apt-get install -y build-essential cmake git pkg-config net-tools \
&& apt-get install -y libfreetype6 libfreetype6-dev \
&& apt-get install -y wget unzip libpq-dev gfortran \
&& apt-get install -y libffi-dev libssl-dev libxml2-dev libxslt1-dev \
&& apt-get install -y libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev \
&& apt-get install -y libhdf5-serial-dev libleveldb-dev liblmdb-dev libprotobuf-dev \
&& apt-get install -y libsnappy-dev protobuf-compiler \
&& apt-get install -y libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
&& apt-get install -y libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev \
&& apt-get install -y libopencv-dev libopenblas-dev liblapack-dev graphviz

# ADD BAZEL REPO
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk bazel \
&& apt-get upgrade bazel -y

RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

ENV PYTHON_VERSION 2.7
RUN apt-get install -y python$PYTHON_VERSION python$PYTHON_VERSION-dev \
&& ln -sf /usr/bin/python2.7 /usr/bin/python \
&& wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py \
&& pip install --upgrade pip \
&& pip install numpy matplotlib \
&& pip install opencv-python graphviz

# TENSORFLOW
RUN pip install tensorflow-gpu

# CAFFE
ENV CAFFE_CLONE_TAG=master
ENV CAFFE_ROOT=${INSTALL_ROOT}/caffe
ENV CAFFE_GIT_REPO=https://github.com/BVLC/caffe
WORKDIR $CAFFE_ROOT
RUN git clone -b ${CAFFE_CLONE_TAG} --depth 1 ${CAFFE_GIT_REPO} . && \
    pip install -r python/requirements.txt && \
    mkdir build && cd build && \
    cmake .. && \
    make -j${NUM_CORES}

# MXNET
ENV MXNET_CLONE_TAG=v0.12.0
ENV MXNET_ROOT=${INSTALL_ROOT}/mxnet
ENV MXNET_GIT_REPO=https://github.com/apache/incubator-mxnet
WORKDIR $MXNET_ROOT
RUN git clone --recursive -b ${MXNET_CLONE_TAG} --depth 1 ${MXNET_GIT_REPO} . && \
    make -j${NUM_CORES} USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
# To remove the following error caused by opencv
# libdc1394 error: Failed to initialize libdc1394"
CMD sh -c 'ln -s /dev/null /dev/raw1394'; bash
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install some dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y ca-certificates curl \
    && apt-get install -y apt-utils bc build-essential cmake curl g++ gfortran git \
    && apt-get install -y libffi-dev libfreetype6 libfreetype6-dev libhdf5-dev libjpeg-dev liblcms2-dev \
    && apt-get install -y libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev \
    && apt-get install -y libhdf5-serial-dev libleveldb-dev liblmdb-dev libprotobuf-dev \
    && apt-get install -y libopenblas-dev liblapack-dev libpng12-dev libssl-dev libxml2-dev libxslt1-dev \
    && apt-get install -y libwebp-dev libzmq3-dev pkg-config python-dev software-properties-common \
    && apt-get install -y unzip vim wget zlib1g-dev libvtk6-dev zlib1g-dev libwebp-dev \
    && apt-get install -y libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev libdc1394-22-dev \
    && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev \
    && apt-get install -y libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev libtbb-dev \
    && apt-get install -y libeigen3-dev python-dev python-tk python-numpy python3-dev python3-tk python3-numpy ant default-jdk \
    && apt-get install -y libopencv-dev graphviz \
    && apt-get install -y doxygen libsnappy-dev protobuf-compiler \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* \
    # Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
    && update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

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
    && pip --no-cache-dir install --upgrade pip \
    && pip --no-cache-dir install numpy matplotlib jupyter six wheel Pillow scipy h5py \
    && pip --no-cache-dir install opencv-python graphviz scikit-image pandas sympy scikit-learn ipython \
    && pip --no-cache-dir install tensorflow-tensorboard seaborn joblib line-profiler dask boto botocore beautifulsoup4 bs4 \
    && pip --no-cache-dir install lmdb lxml protobuf SQLAlchemy  
 
# Install TensorFlow
RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@2.1.2

# Install MXNET
RUN git clone --recursive https://github.com/apache/incubator-mxnet \
    && cd incubator-mxnet \
    && make -j4 USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 \
    && cd python \
    && pip install -e . \
    && cd

# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888
# To remove the following error caused by opencv
# libdc1394 error: Failed to initialize libdc1394"
CMD sh -c 'ln -s /dev/null /dev/raw1394'; bash

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r//' /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR "/root"
ENTRYPOINT ["/entrypoint.sh"]

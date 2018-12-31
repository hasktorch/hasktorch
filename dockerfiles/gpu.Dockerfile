FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
MAINTAINER Junji Hashimoto "junji.hashimoto@gree.net"
RUN apt-get update -qq && apt-get -y --allow-downgrades --allow-remove-essential --allow-change-held-packages install locales software-properties-common apt-transport-https
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
RUN add-apt-repository ppa:hvr/ghc
RUN apt-get update -qq && apt-get -y --allow-downgrades --allow-remove-essential --allow-change-held-packages install \
  curl git build-essential libtinfo-dev libssl-dev zlib1g-dev \
  liblapack-dev libblas-dev \
  ghc-8.4.4 cabal-install-head \
  devscripts debhelper \
  cmake python3-pip python3-yaml

RUN curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB > GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list
RUN apt-get update -qq && apt-get -y --allow-downgrades --allow-remove-essential --allow-change-held-packages install \
  intel-mkl-64bit-2019.1-053

ENV PATH /opt/ghc/bin:$PATH
ENV LD_LIBRARY_PATH /hasktorch/ffi/deps/aten/build/lib:$LD_LIBRARY_PATH
RUN cabal new-update
RUN git clone --recursive https://github.com/hasktorch/hasktorch.git
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN cd hasktorch/ffi/deps && ./build-aten.sh && cd ../../..
RUN cd hasktorch && make init && cd ..
RUN cd hasktorch && cabal new-build all && cd ..
RUN cd hasktorch && cabal new-run gaussian-process && cd ..

RUN mkdir -p /tmp/app/debian
COPY gpu.Makefile /tmp/app/Makefile
COPY debian /tmp/app/debian
RUN cd /tmp/app && dch --create --package libaten-dev -v 0.1.0 'This package is built with CUDA support' && cd ../..

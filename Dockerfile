FROM nvidia/cuda:10.1-devel-ubuntu18.04

WORKDIR /hasktorch

RUN apt update -qq
RUN apt -y --allow-downgrades --allow-remove-essential --allow-change-held-packages install locales software-properties-common apt-transport-https
RUN add-apt-repository -y ppa:hvr/ghc
RUN apt update -qq
RUN apt -y purge ghc* cabal-install* || true
RUN apt -y --allow-downgrades --allow-remove-essential --allow-change-held-packages install build-essential zlib1g-dev liblapack-dev libblas-dev ghc-8.6.5 cabal-install-3.0 devscripts debhelper python3-pip cmake curl wget unzip git libtinfo-dev python3 python3-yaml

COPY . /hasktorch

ENV PATH="/opt/ghc/bin:${PATH}"
RUN cd deps/ && ./get-deps.sh -a cu101
ENV LD_LIBRARY_PATH="/hasktorch/deps/libtorch/lib/:/hasktorch/deps/mklml/lib/:/hasktorch/deps/libtokenizers/lib/:${LD_LIBRARY_PATH}"
RUN ./setup-cabal.sh
RUN cabal v2-update
RUN cabal v2-install hspec-discover
RUN cabal v2-build all --jobs=2 --write-ghc-environment-files=always

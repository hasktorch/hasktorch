#!/usr/bin/env bash

# - Retrieves a prebuilt libtorch binary per https://pytorch.org/cppdocs/installing.html
# - Retrieves a release binary for mkl https://github.com/intel/mkl-dnn/releases
#   which is a runtime dependency that is not package w/ libtorch

set -eu

case "$(uname)" in
  "Darwin")
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.0.0.zip
    unzip libtorch-macos-1.0.0.zip
    rm libtorch-macos-1.0.0.zip
    wget https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_mac_2019.0.1.20181227.tgz
    tar -xzf mklml_mac_2019.0.1.20181227.tgz
    rm -f mklml_mac_2019.0.1.20181227.tgz
    rm -f mklml_mac_2019.0.1.20181227.tgz.1
    ;;
  "Linux")
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    unzip libtorch-shared-with-deps-latest.zip
    rm libtorch-shared-with-deps-latest.zip
    wget https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_lnx_2019.0.1.20181227.tgz
    tar -xzf mklml_lnx_2019.0.1.20181227.tgz
    rm -f mklml_lnx_2019.0.1.20181227.tgz
    rm -f mklml_lnx_2019.0.1.20181227.tgz.1
    ;;
esac

#!/usr/bin/env bash

# - Retrieves a prebuilt libtorch binary per https://pytorch.org/cppdocs/installing.html
# - Retrieves a release binary for mkl https://github.com/intel/mkl-dnn/releases
#   which is a runtime dependency that is not package w/ libtorch

set -eu

case "$(uname)" in
  "Darwin")
    USE_BINARY_FOR_CI="$1"
    if [ -z "$USE_BINARY_FOR_CI" ] ; then
      wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.1.0.zip
      unzip libtorch-macos-1.1.0.zip
      rm libtorch-macos-1.1.0.zip
    else
      wget https://github.com/hasktorch/libtorch-binary-for-ci/releases/download/1.1.0/cpu-libtorch-macos-latest.zip
      unzip cpu-libtorch-macos-latest.zip
      rm cpu-libtorch-macos-latest.zip
    fi
    wget https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_mac_2019.0.1.20181227.tgz
    tar -xzf mklml_mac_2019.0.1.20181227.tgz
    rm -f mklml_mac_2019.0.1.20181227.tgz
    rm -f mklml_mac_2019.0.1.20181227.tgz.1
    rm -rf mklml
    mv mklml_mac_2019.0.1.20181227 mklml
    ;;
  "Linux")
    USE_BINARY_FOR_CI="$1"
    if [ -z "$USE_BINARY_FOR_CI" ] ; then
      wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
      unzip libtorch-shared-with-deps-latest.zip
      rm libtorch-shared-with-deps-latest.zip
    else
      wget https://github.com/hasktorch/libtorch-binary-for-ci/releases/download/1.1.0/cpu-libtorch-shared-with-deps-latest.zip
      unzip cpu-libtorch-shared-with-deps-latest.zip
      rm cpu-libtorch-shared-with-deps-latest.zip
    fi
    wget https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_lnx_2019.0.1.20181227.tgz
    tar -xzf mklml_lnx_2019.0.1.20181227.tgz
    rm -f mklml_lnx_2019.0.1.20181227.tgz
    rm -f mklml_lnx_2019.0.1.20181227.tgz.1
    rm -rf mklml
    mv mklml_lnx_2019.0.1.20181227 mklml
    ln -s libmklml_intel.so mklml/lib/libmklml.so
    ;;
esac

# Following codes are copied from pytorch/tools/run-clang-tidy-in-ci.sh.
# Generate ATen files.
pushd pytorch

if [[ ! -d build ]]; then
mkdir build
fi

python aten/src/ATen/gen.py \
  -s aten/src/ATen \
  -d build/aten/src/ATen \
  aten/src/ATen/Declarations.cwrap \
  aten/src/THNN/generic/THNN.h \
  aten/src/THCUNN/generic/THCUNN.h \
  aten/src/ATen/nn.yaml \
  aten/src/ATen/native/native_functions.yaml

# Sanitize "name: n" fields to be strings rather than booleans in Declarations.yaml

case "$(uname)" in
  "Darwin")
    sed -i '' -e "s/ name: n$/ name: 'n'/g" -e "s/ name: N$/ name: 'N'/g" build/aten/src/ATen/Declarations.yaml
    ;;
  "Linux")
    sed -i -e "s/ name: n$/ name: 'n'/g" -e "s/ name: N$/ name: 'N'/g" build/aten/src/ATen/Declarations.yaml
    ;;
esac

popd

#!/usr/bin/env bash

# - gets submodules recursively to get pytorch and inline-c fork repo dependencies
# - Retrieves a prebuilt libtorch binary per https://pytorch.org/cppdocs/installing.html
# - Retrieves a release binary for mkl https://github.com/intel/mkl-dnn/releases
#   which is a runtime dependency that is not package w/ libtorch

set -eu

usage_exit() {
    echo "Usage: $0 [-n] [-c] [-a "cpu" or "cu92" or "cu101" or "cu102"] [-s]" 1>&2
    echo " -n # Use nightly libtorch w/  -l" 1>&2
    echo "    # Use libtorch-1.5.0   w/o -l" 1>&2
    echo "" 1>&2
    echo " -c # Download libtorch from hasktorch's site w/ -c" 1>&2
    echo "    # Download libtorch from pytorch's site w/o  -c" 1>&2
    echo "" 1>&2
    echo " -a cpu   # Use CPU without CUDA" 1>&2
    echo " -a cu92  # Use CUDA-9" 1>&2
    echo " -a cu101 # Use CUDA-10.1" 1>&2
    echo " -a cu102 # Use CUDA-10.2" 1>&2
    echo "" 1>&2
    echo " -s # Skip download" 1>&2
    echo "" 1>&2
    echo " -h # Show this help" 1>&2
    exit 1
}

USE_NIGHTLY=0
USE_BINARY_FOR_CI=0
COMPUTE_ARCH=cpu
SKIP_DOWNLOAD=0
VERSION=1.5.0

while getopts nca:sh OPT
do
    case $OPT in
        n)  USE_NIGHTLY=1
            ;;
        c)  USE_BINARY_FOR_CI=1
            ;;
        a)  COMPUTE_ARCH=$OPTARG
            ;;
        s)  SKIP_DOWNLOAD=1
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done

if [ "$SKIP_DOWNLOAD" = 0 ] ; then
  # git submodule update --init --recursive
  # pytorch v1.3 has unlinked submodule of https://github.com/IvanKobzarev/fbjni.git
  # For now, we can not update recursively.
  git submodule update --init

  case "$(uname)" in
    "Darwin")
      if [ "$USE_NIGHTLY" = 1 ] ; then
        wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip
        unzip libtorch-macos-latest.zip
        rm libtorch-macos-latest.zip
      elif [ "$USE_BINARY_FOR_CI" = 1 ] ; then
        wget https://github.com/hasktorch/libtorch-binary-for-ci/releases/download/${VERSION}/cpu-libtorch-macos-latest.zip
        unzip cpu-libtorch-macos-latest.zip
        rm cpu-libtorch-macos-latest.zip
      else
        wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip
        unzip libtorch-macos-${VERSION}.zip
        rm libtorch-macos-${VERSION}.zip
      fi
      wget https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_mac_2019.0.1.20181227.tgz
      tar -xzf mklml_mac_2019.0.1.20181227.tgz
      rm -f mklml_mac_2019.0.1.20181227.tgz
      rm -f mklml_mac_2019.0.1.20181227.tgz.1
      rm -rf mklml
      mv mklml_mac_2019.0.1.20181227 mklml
      ;;
    "Linux")
      if [ "$USE_NIGHTLY" = 1 ] ; then
        wget https://download.pytorch.org/libtorch/nightly/${COMPUTE_ARCH}/libtorch-cxx11-abi-shared-with-deps-latest.zip
        unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
        rm libtorch-cxx11-abi-shared-with-deps-latest.zip
      elif [ "$USE_BINARY_FOR_CI" = 1 ] ; then
        wget https://github.com/hasktorch/libtorch-binary-for-ci/releases/download/${VERSION}/${COMPUTE_ARCH}-libtorch-cxx11-abi-shared-with-deps-latest.zip
        unzip ${COMPUTE_ARCH}-libtorch-cxx11-abi-shared-with-deps-latest.zip
        rm ${COMPUTE_ARCH}-libtorch-cxx11-abi-shared-with-deps-latest.zip
      else
	case "${COMPUTE_ARCH}" in
	      "cpu" )   URL=https://download.pytorch.org/libtorch/${COMPUTE_ARCH}/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Bcpu.zip ;;
	      "cu92" ) URL=https://download.pytorch.org/libtorch/${COMPUTE_ARCH}/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Bcu92.zip ;;
	      "cu101" )   URL=https://download.pytorch.org/libtorch/${COMPUTE_ARCH}/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Bcu101.zip ;;
	      "cu102" )   URL=https://download.pytorch.org/libtorch/${COMPUTE_ARCH}/libtorch-cxx11-abi-shared-with-deps-${VERSION}.zip ;;
	esac
	wget -O libtorch-cxx11-abi-shared-with-deps-${VERSION}.zip "$URL"
        unzip libtorch-cxx11-abi-shared-with-deps-${VERSION}.zip
        rm libtorch-cxx11-abi-shared-with-deps-${VERSION}.zip
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
fi

# Following codes are copied from pytorch/tools/run-clang-tidy-in-ci.sh.
# Generate ATen files.

echo "Generate ATen files."
pushd pytorch

if [[ ! -d build ]]; then
mkdir build
fi

PYTHON=python
if ! (python --version | grep "Python 2") ;then
    PYTHON=python3
fi

$PYTHON aten/src/ATen/gen.py \
  -s aten/src/ATen \
  -d build/aten/src/ATen \
  aten/src/ATen/Declarations.cwrap \
  aten/src/THCUNN/generic/THCUNN.h \
  aten/src/ATen/nn.yaml \
  aten/src/ATen/native/native_functions.yaml


# Sanitize "name: n" fields to be strings rather than booleans in Declarations.yaml

case "$(uname)" in
  "Darwin")
    sed -i '' -e "s/ name: n$/ name: 'n'/g" -e "s/ name: N$/ name: 'N'/g" build/aten/src/ATen/Declarations.yaml
    sed -i '' -e "s/ name: t$/ name: 't'/g" -e "s/ name: T$/ name: 'T'/g" build/aten/src/ATen/Declarations.yaml
    sed -i '' -e "s/ name: y$/ name: 'y'/g" -e "s/ name: Y$/ name: 'Y'/g" build/aten/src/ATen/Declarations.yaml
    sed -i '' -e "s/ default: \([^'].*\)$/ default: '\1'/g" build/aten/src/ATen/Declarations.yaml
    ;;
  "Linux")
    sed -i -e "s/ name: n$/ name: 'n'/g" -e "s/ name: N$/ name: 'N'/g" build/aten/src/ATen/Declarations.yaml
    sed -i -e "s/ name: t$/ name: 't'/g" -e "s/ name: T$/ name: 'T'/g" build/aten/src/ATen/Declarations.yaml
    sed -i -e "s/ name: y$/ name: 'y'/g" -e "s/ name: Y$/ name: 'Y'/g" build/aten/src/ATen/Declarations.yaml
    sed -i -e "s/ default: \([^'].*\)$/ default: '\1'/g" build/aten/src/ATen/Declarations.yaml
    ;;
esac

popd

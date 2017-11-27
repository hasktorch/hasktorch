#!/usr/bin/env bash
set -eu

mkdir -p aten-spec/

# Sanitize cwrap into a yaml-compliant format

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/TensorApply.cwrap > aten-spec/TensorApply.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/TensorMethods.cwrap > aten-spec/TensorMethods.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/cudnn/cuDNN.cwrap > aten-spec/cuDNN.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/TensorSerialization.cwrap > aten-spec/TensorSerialization.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/SparseTensor.cwrap > aten-spec/SparseTensor.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/Tensor.cwrap > aten-spec/Tensor.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/TensorCompare.cwrap > aten-spec/TensorCompare.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/TensorCuda.cwrap > aten-spec/TensorCuda.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/TensorMath.cwrap > aten-spec/TensorMath.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/torch/csrc/generic/methods/TensorRandom.cwrap > aten-spec/TensorRandom.yaml

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/aten/src/ATen/Declarations.cwrap > aten-spec/Declarations.yaml

head -50 aten-declarations.yaml

printf ".\n.\n.\n\nWrote ATen spec files"

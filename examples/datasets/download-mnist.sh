#!/usr/bin/env bash

mkdir mnist
pushd mnist
#MNIST_HOST=http://yann.lecun.com/exdb/mnist
MNIST_HOST=https://ossci-datasets.s3.amazonaws.com/mnist
wget $MNIST_HOST/train-images-idx3-ubyte.gz
wget $MNIST_HOST/train-labels-idx1-ubyte.gz
wget $MNIST_HOST/t10k-images-idx3-ubyte.gz
wget $MNIST_HOST/t10k-labels-idx1-ubyte.gz
popd


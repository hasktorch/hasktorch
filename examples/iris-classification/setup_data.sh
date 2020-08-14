#!/usr/bin/env bash
mkdir data
pushd data
# wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
popd

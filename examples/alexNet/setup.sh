#!/usr/bin/env bash

wget https://github.com/hasktorch/hasktorch-artifacts/releases/download/alexnet/alexNet.pt
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xf images.tar.gz
rm images.tar.gz

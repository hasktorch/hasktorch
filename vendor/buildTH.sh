#!/bin/bash -eu

mkdir -p ./build
rm -rf build/*
cd ./build; cmake ../TH/CMakeLists.txt -B. -DCMAKE_CXX_COMPILER=g++-6 -DCMAKE_CC_COMPILER=gcc-6 -DCXX=g++-6 -DCC=gcc-6; make

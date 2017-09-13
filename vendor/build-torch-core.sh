#!/bin/bash -eu

mkdir -p ./build
# rm -rf build/*

if [ "$(uname)" == "Darwin" ]; then
    if ! [ -x "$(command -v gcc-6)" ]; then
        echo 'Error: gcc-6 is not installed, use homebrew to install it.' >&2
        exit 1
    fi
    echo "Running as OSX ..."
    CXX=g++-6
    CC=gcc-6
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    if ! [ -x "$(command -v gcc)" ]; then
        echo 'Error: gcc is not installed.' >&2
        exit 1
    fi
    echo "Running as Linux ..."
    CXX=g++
    CC=gcc
else
    echo "Unknown OS"
    exit
fi

echo "Compilers:"
echo "  $CXX"
echo "  $CC"

cd ./build; cmake ../TH/CMakeLists.txt -B. -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CC_COMPILER=$CC -DCXX=$CXX -DCC=$CC; make

#!/bin/bash -eu

mkdir -p ./build
cd ./build; cmake ../TH/CMakeLists.txt -B.; make

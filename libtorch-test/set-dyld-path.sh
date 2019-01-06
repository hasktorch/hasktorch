#!/usr/bin/env bash

# - Sets LD_LIBRARY_PATH (run this using `source`)

export LD_LIBRARY_PATH=$(pwd)/../deps/mklml/lib:$(pwd)/../deps/libtorch/lib:$LD_LIBRARY_PATH 

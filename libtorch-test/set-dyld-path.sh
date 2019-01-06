#!/usr/bin/env bash

# - Sets LD_LIBRARY_PATH (run this using `source`)

# case "$(uname)" in
#   "Darwin")
#     export LD_LIBRARY_PATH=$(pwd)/../deps/mklml_mac_2019.0.1.20181227/lib:$(pwd)/../deps/libtorch/lib:$LD_LIBRARY_PATH
#     ;;
#   "Linux")
#     export LD_LIBRARY_PATH=$(pwd)/../deps/mklml_lnx_2019.0.1.20181227/lib:$(pwd)/../deps/libtorch/lib:$LD_LIBRARY_PATH
#     ;;
# esac

export LD_LIBRARY_PATH=$(pwd)/../deps/mklml/lib:$(pwd)/../deps/libtorch/lib:$LD_LIBRARY_PATH 

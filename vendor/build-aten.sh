#!/usr/bin/env bash
#set -eu

function check_bin {
  local BIN="$1"
  if ! [ -x "$(command -v $BIN)" ]; then
      echo "Error: $BIN is not installed." >&2
      exit 1
  fi
}

case "$(uname)" in
  "Darwin")
    echo "Running as OSX"
    check_bin "gcc-7"
    CXX=g++-7
    CC=gcc-7
    ;;

  "Linux")
    if uname -v | grep "Ubuntu" &> /dev/null && lsb_release -a 2>&1 | grep " 17." &> /dev/null; then
      echo "Running with gcc-4.8 on Ubuntu 17.xx -- see https://github.com/pytorch/pytorch/issues/5136 for details"
      check_bin "gcc-4.8"
      CXX=g++-4.8
      CC=gcc-4.8
    else
      echo "Running as Linux with default gcc."
      check_bin "gcc"
      CXX=g++
      CC=gcc
    fi
    ;;

  "FreeBSD")
    echo "Running as FreeBSD with default gcc."
    check_bin "gcc"
    CXX=g++
    CC=gcc
    ;;
  *)
    echo "Unknown OS"
    exit 1
    ;;
  esac

function build {
  echo "Compilers:"
  echo "  $CXX"
  echo "  $CC"

  mkdir -p ./aten/build
  cd ./aten/build
  if nvcc --version &> /dev/null; then
    with_cuda=true
  else
    with_cuda=false
  fi
  cmake .. -DNO_CUDA=$with_cuda -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CC_COMPILER=$CC -DCXX=$CXX -DCC=$CC -Wno-dev
  make install
}

function install {
  mkdir build
  cp ./aten/build/src/ATen/libATen.* ./build/
}

case "$1" in
  ""|"all")
    build
    install
    ;;
  "build")
    build
    ;;
  "install")
    install
    ;;
  "clean")
    rm -rf ./aten/build
    ;;
  "help"|*)
    echo "./build-aten.sh -- build the aten library"
    echo ""
    echo "COMMANDS"
    echo ""
    echo "  all     - build, then install. Default command."
    echo "  build   - build the aten library"
    echo "  install - copy built binaries to the ./build folder"
    echo "  help    - show this prompt"
esac

# if [ -x "$(command -v nm)" ]; then
#     echo "Checking symbols in dylib:"
#     nm -gU ./build/libTH.dylib
#     exit 1
# fi

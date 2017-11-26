#!/usr/bin/env bash
set -eu

mkdir -p ./build

case "$(uname)" in
  "Darwin")
    if ! [ -x "$(command -v gcc-6)" ]; then
        echo 'Error: gcc-6 is not installed, use homebrew to install it.' >&2
        exit 1
    fi
    echo "Running as OSX ..."
    CXX=g++-6
    CC=gcc-6
    ;;

  "Linux")
    if ! [ -x "$(command -v gcc)" ]; then
        echo 'Error: gcc is not installed.' >&2
        exit 1
    fi
    echo "Running as Linux ..."
    CXX=g++
    CC=gcc
    ;;

  "FreeBSD")
    if ! [ -x "$(command -v gcc)" ]; then
        echo 'Error: gcc is not installed.' >&2
        exit 1
    fi
    echo "Running as FreeBSD..."
    CXX=g++
    CC=gcc
    ;;
  *)
    echo "Unknown OS"
    exit 1
    ;;
  esac

echo "Compilers:"
echo "  $CXX"
echo "  $CC"

cd ./build; cmake ../aten/CMakeLists.txt -B. -DNO_CUDA=true -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CC_COMPILER=$CC -DCXX=$CXX -DCC=$CC; make; cd ..
cp ./build/src/ATen/libATen.* ./build/

# if [ -x "$(command -v nm)" ]; then
#     echo "Checking symbols in dylib:"
#     nm -gU ./build/libTH.dylib
#     exit 1
# fi

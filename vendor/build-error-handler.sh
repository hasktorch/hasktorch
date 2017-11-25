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
        EXTENSION=dylib
        DLARG=-dynamiclib
        ;;

    "Linux")
        if ! [ -x "$(command -v gcc)" ]; then
            echo 'Error: gcc is not installed.' >&2
            exit 1
        fi
        echo "Running as Linux ..."
        CXX=g++
        CC=gcc
        EXTENSION=so
        DLARG=-shared
        ;;

    "FreeBSD")
        if ! [ -x "$(command -v gcc)" ]; then
            echo 'Error: gcc is not installed.' >&2
            exit 1
        fi
        echo "Running as FreeBSD..."
        CXX=g++
        CC=gcc
        EXTENSION=so
        DLARG=-shared
        ;;
    *)

        echo "Unknown OS"
        exit 1
        ;;
esac

echo "Compilers:"
echo "  $CXX"
echo "  $CC"

rm -f ./libEH.o ./libEH.$EXTENSION ./build/libEH.$EXTENSION
$CXX -Wall -c error_handler.cpp -fno-common -o libEHX.o
$CXX $DLARG ./libEHX*.o -o libEHX.$EXTENSION
rm -f libEHX.o
mv libEHX.$EXTENSION ./build/

# if [ -x "$(command -v nm)" ]; then
#     echo "Checking symbols in dylib:"
#     nm -gU ./build/libEHX.dylib
#     exit 1
# fi

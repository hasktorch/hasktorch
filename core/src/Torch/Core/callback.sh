#!/usr/bin/env bash

ghc -c -O ErrorCallback.hs

gcc -I/usr/local/lib/ghc-8.0.2/include/ test_callback.c ErrorCallback.o -o test_callback.o

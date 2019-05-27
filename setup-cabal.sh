#!/bin/bash

set -xe

curl https://www.stackage.org/lts-13.23/cabal.config > cabal.project.freeze

cat <<EOF > cabal.project.local

extra-include-dirs:
    $(pwd)/deps/libtorch/include/torch/csrc/api/include
  , $(pwd)/deps/libtorch/include

extra-lib-dirs:
    $(pwd)/deps/libtorch/lib
  , $(pwd)/deps/mklml/lib

EOF

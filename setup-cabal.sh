#!/usr/bin/env bash

set -xe

curl https://www.stackage.org/lts-14.7/cabal.config | \
sed -e 's/inline-c ==.*,/inline-c ==0.9.0.0,/g' -e 's/inline-c-cpp ==.*,/inline-c-cpp ==0.4.0.0,/g' > cabal.project.freeze

cat <<EOF > cabal.project.local

package libtorch-ffi
  extra-include-dirs: $(pwd)/deps/libtorch/include/torch/csrc/api/include
  extra-include-dirs: $(pwd)/deps/libtorch/include
  extra-lib-dirs: $(pwd)/deps/libtorch/lib

package *
  extra-lib-dirs: $(pwd)/deps/mklml/lib

EOF

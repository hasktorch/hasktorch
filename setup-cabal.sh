#!/usr/bin/env bash

set -xe

if ghc --version | grep 8.10 ; then
ln -s cabal.project.freeze.ghc810 cabal.project.freeze
else
curl https://www.stackage.org/lts-15.0/cabal.config | \
sed -e 's/inline-c ==.*,/inline-c ==0.9.0.0,/g' \
    -e 's/inline-c-cpp ==.*,/inline-c-cpp ==0.4.0.0,/g' \
    -e 's/ghc-typelits-natnormalise ==.*,/ghc-typelits-natnormalise ==0.7,/g' \
    -e 's/ghc-typelits-knownnat ==.*,/ghc-typelits-knownnat ==0.7,/g' \
    -e 's/ghc-typelits-extra ==.*,/ghc-typelits-extra ==0.3.2,/g' \
    > cabal.project.freeze
fi


cat <<EOF > cabal.project.local

package libtorch-ffi
  extra-include-dirs: $(pwd)/deps/libtorch/include/torch/csrc/api/include
  extra-include-dirs: $(pwd)/deps/libtorch/include
  extra-lib-dirs: $(pwd)/deps/libtorch/lib

package *
  extra-lib-dirs: $(pwd)/deps/mklml/lib

EOF

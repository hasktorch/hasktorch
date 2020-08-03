#!/usr/bin/env bash

set -xe

if ghc --version | grep 8.10 ; then
ln -fs cabal.project.freeze.ghc810 cabal.project.freeze
else
curl https://www.stackage.org/lts-15.0/cabal.config | \
sed -e 's/inline-c ==.*,/inline-c ==0.9.0.0,/g' \
    -e 's/inline-c-cpp ==.*,/inline-c-cpp ==0.4.0.0,/g' \
    -e 's/ghc-typelits-natnormalise ==.*,/ghc-typelits-natnormalise ==0.7,/g' \
    -e 's/ghc-typelits-knownnat ==.*,/ghc-typelits-knownnat ==0.7,/g' \
    -e 's/ghc-typelits-extra ==.*,/ghc-typelits-extra ==0.3.2,/g' \
    > cabal.project.freeze
fi

case "$(uname)" in
  "Darwin")
      TOTAL_MEM_GB=`sysctl hw.physmem | awk '{print int($2/1024/1024/1024)}'`
      NUM_CPU=$(sysctl -n hw.ncpu)
    ;;
  "Linux")
      TOTAL_MEM_GB=`grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}'`
      NUM_CPU=$(nproc --all)
    ;;
esac

USED_MEM_GB=`echo $TOTAL_MEM_GB | awk '{print int(($1 + 1) / 2)}'`
USED_NUM_CPU=`echo $NUM_CPU | awk '{print int(($1 + 1) / 2)}'`
USED_NUM_CPU=`echo $USED_MEM_GB $USED_NUM_CPU | awk '{if($1<x$2) {print $1} else {print $2}}'`
USED_MEM_GB=`echo $USED_NUM_CPU | awk '{print ($1)"G"}'`
USED_MEMX2_GB=`echo $USED_NUM_CPU | awk '{print ($1 * 2)"G"}'`


cat <<EOF > cabal.project.local

package libtorch-ffi
  extra-include-dirs: $(pwd)/deps/libtorch/include/torch/csrc/api/include
  extra-include-dirs: $(pwd)/deps/libtorch/include
  extra-lib-dirs: $(pwd)/deps/libtorch/lib

package *
  extra-lib-dirs: $(pwd)/deps/mklml/lib

package libtorch-ffi
    ghc-options: -j${USED_NUM_CPU} +RTS -A128m -n2m -M${USED_MEM_GB} -RTS

package hasktorch
    ghc-options: -j${USED_NUM_CPU} +RTS -A128m -n2m -M${USED_MEMX2_GB} -RTS

EOF

#!/usr/bin/env bash

set -xe
ghc --version

#curl https://www.stackage.org/lts-21.25/cabal.config |\

curl https://www.stackage.org/lts-22.44/cabal.config | \
sed -e 's/with-compiler: .*$//g' |\
sed -e 's/.*inline-c.*//g' > cabal.project.freeze

case "$(uname)" in
  "Darwin")
      TOTAL_MEM_GB=`sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}'`
      NUM_CPU=$(sysctl -n hw.ncpu)
      GHC_OPTIONS="-optl-ld_classic"
    ;;
  "Linux")
      TOTAL_MEM_GB=`grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}'`
      NUM_CPU=$(nproc --all)
      GHC_OPTIONS=""
    ;;
esac

USED_MEM_GB=`echo $TOTAL_MEM_GB | awk '{print int(($1 + 1) / 2)}'`
USED_NUM_CPU=`echo $NUM_CPU | awk '{print int(($1 + 1) / 2)}'`
USED_NUM_CPU=`echo $USED_MEM_GB $USED_NUM_CPU | awk '{if($1<$2) {print $1} else {print $2}}'`
USED_MEM_GB=`echo $USED_NUM_CPU | awk '{print ($1)"G"}'`
USED_MEMX2_GB=`echo $USED_NUM_CPU | awk '{print ($1 * 2)"G"}'`

cat <<EOF > cabal.project.local

package libtorch-ffi
  extra-include-dirs: $XDG_CACHE_HOME/libtorch//include/torch/csrc/api/include
  extra-include-dirs: $XDG_CACHE_HOME/libtorch/include
  extra-lib-dirs: $XDG_CACHE_HOME/libtorch/lib
  extra-include-dirs: /opt/homebrew/include
  extra-lib-dirs: /opt/homebrew/lib
  extra-lib-dirs: /opt/homebrew/opt/libomp/lib

package *
  extra-lib-dirs: $XDG_CACHE_HOME/libtorch/lib
  extra-lib-dirs: $XDG_CACHE_HOME/libtokenizers/lib
  extra-lib-dirs: /opt/homebrew/lib
  extra-lib-dirs: /opt/homebrew/opt/libomp/lib

package libtorch-ffi
    ghc-options: ${GHC_OPTIONS} -j${USED_NUM_CPU} +RTS -A128m -n2m -M${USED_MEM_GB} -RTS

package hasktorch
    ghc-options: -j${USED_NUM_CPU} +RTS -A128m -n2m -M${USED_MEMX2_GB} -RTS

package vector
    ghc-options: -j${USED_NUM_CPU} +RTS -A128m -n2m -M${USED_MEMX2_GB} -RTS

EOF

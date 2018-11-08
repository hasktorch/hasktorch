#!/usr/bin/env bash -eu

echo "
extra-lib-dirs: $PWD/ffi/deps/aten/build/lib/

extra-include-dirs: $PWD/ffi/deps/aten/build/include/
extra-include-dirs: $PWD/ffi/deps/aten/build/include/TH
extra-include-dirs: $PWD/ffi/deps/aten/build/include/THNN
extra-include-dirs: $PWD/ffi/deps/aten/build/include/THC
extra-include-dirs: $PWD/ffi/deps/aten/build/include/THCUNN

package *
  flags: -cuda +lite
" > cabal.project.local

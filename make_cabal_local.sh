#!/usr/bin/env bash

set -eu

printf "\nCreating a cabal.project.local file.\n\n"
printf "This script should be run from the top-level hasktorch directory.\n\n"

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

printf "Created cabal.project.local with local include/lib paths containing:\n\n"
printf "========================================\n"
cat cabal.project.local
printf "========================================\n\n"
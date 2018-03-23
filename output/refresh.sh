#!/usr/bin/env bash
set -eu

function nuke {
  lib=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  rm -rf ./raw/${lib}/src/
  (
    cd ..
    cabal new-run hasktorch-codegen:ht-codegen -- --lib $1 --type concrete
    cabal new-run hasktorch-codegen:ht-codegen -- --lib $1 --type generic
  )
  rm -rf ../raw/${lib}/src/
  rsync -arv {.,..}/raw/${lib}/src/
}

nuke TH
nuke THC
nuke THNN
nuke THCUNN


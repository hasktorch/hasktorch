#!/usr/bin/env bash
set -eu

function __nuke {
  lib=$1
  src=$2
  out=$3
  rm -rf $src
  (
    cd ..
    cabal new-run hasktorch-codegen:ht-codegen -- --lib $lib --type concrete
    cabal new-run hasktorch-codegen:ht-codegen -- --lib $lib --type generic
  )
  rm -rf $out
  rsync -arv $src $out
}

function nuke {
  lib=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  __nuke $1 {.,..}/raw/${lib}/src/
}

function nuke_nn {
  libnn=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case $libnn in
    "thnn") lib="th" ;;
    "thcunn") lib="thc" ;;
    *) echo "$libnn is not a valid nn library"; exit 1 ;;
  esac

  __nuke $1 ./raw/${libnn}/src/ ../raw/${lib}/nn/
}


nuke TH
nuke THC

nuke_nn THNN
nuke_nn THCUNN


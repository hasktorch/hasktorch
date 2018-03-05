#!/usr/bin/env bash

echo "Recommended to run this script using `source ./codegen-and-rebuild.sh` to evaluate it in the current shell"

set -eu

cabal new-build hasktorch-codegen || {
  echo "can't build hasktorch-codegen, exiting early"
  exit 1
}

cabal new-build hasktorch-codegen:ht-codegen -- --lib TH --type concrete
cabal new-build hasktorch-codegen:ht-codegen -- --lib TH --type generic

# use a subshell here to jump back to project root (SC2103)
( cd output || exit 1; ./refresh.sh )

rm -rf dist-newbuild
cabal new-build all

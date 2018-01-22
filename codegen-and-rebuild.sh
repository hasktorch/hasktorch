#!/usr/bin/env bash

echo "Recommended to run this script using `source ./codegen-and-rebuild.sh` to evaluate it in the current shell"

set -eu

stack build hasktorch-codegen || {
  echo "can't build torch-codegen, exiting early"
  exit 1
}

stack exec codegen-concrete
stack exec codegen-generic

# use a subshell here to jump back to project root (SC2103)
( cd output || exit 1; ./refresh.sh )

stack clean
stack build

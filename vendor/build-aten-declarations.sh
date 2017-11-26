#!/usr/bin/env bash
set -eu

# Sanitize cwrap into a yaml-compliant format

sed -e 's/^\[\[$/- function:/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/aten/src/ATen/Declarations.cwrap > declarations.yaml

head -50 declarations.yaml

printf ".\n.\n.\n\nWrote ./declarations.yaml"

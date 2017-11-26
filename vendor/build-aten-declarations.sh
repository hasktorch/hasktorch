#!/usr/bin/env bash
set -eu

# Sanitize cwrap into a yaml-compliant format

sed -e 's/^\[\[$/-/g' \
    -e 's/^\]\]$//g' \
    -e '/^$/d' \
    ./pytorch/aten/src/ATen/Declarations.cwrap > aten-declarations.yaml

head -50 aten-declarations.yaml

printf ".\n.\n.\n\nWrote ./aten-declarations.yaml"

#!/usr/bin/env bash
set -eu

rsync -arv ./raw/th/src/*.hs ../raw/th/src/
rsync -arv ./raw/th/src/generic/*.hs ../raw/th/src/generic/

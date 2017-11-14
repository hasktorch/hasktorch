#!/usr/bin/env bash
set -eu

rsync -arv ./raw/src/*.hs ../raw/src/
rsync -arv ./raw/src/generic/*.hs ../raw/src/generic/

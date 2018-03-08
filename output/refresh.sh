#!/usr/bin/env bash
set -eu

rsync -arv {.,..}/raw/th/src/
rsync -arv {.,..}/raw/thc/src/

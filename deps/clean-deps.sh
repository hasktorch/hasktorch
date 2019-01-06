#!/usr/bin/env bash

set -eu

rm -rf libtorch

# case "$(uname)" in
#   "Darwin")
#     rm -rf mklml_mac_2019.0.1.20181227
#     ;;
#   "Linux")
#     rm -rf mklml_lnx_2019.0.1.20181227
#     ;;
# esac

rm -rf mklml
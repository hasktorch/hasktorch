#!/usr/bin/env bash

set -eu

VERSION=1.11.0

case "$(uname)-$(uname -p)" in
    "Darwin-arm")
	wget -O torch.zip "https://download.pytorch.org/whl/cpu/torch-${VERSION}-cp310-none-macosx_11_0_arm64.whl"
	unzip torch.zip
	ln -s torch libtorch
	;;
esac

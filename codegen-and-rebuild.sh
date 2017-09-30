#!/bin/bash -eu

stack build torch-codegen
stack exec codegen-concrete
stack exec codegen-generic
cd output; ./refresh.sh; cd ..
stack clean
stack build

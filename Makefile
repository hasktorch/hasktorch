init:
	git submodule update --init --recursive
	( cd vendor; ./build-torch-core.sh )
	stack build
	stack test hasktorch-tests

.PHONY: init

init:
	git submodule update --init --recursive
	( cd vendor; ./build-torch-core.sh )
	stack build
	stack test hasktorch-tests

clean:
	stack clean

build: clean
	stack build

refresh:
	rsync -arv ./output/raw/src/*.hs ./raw/src/
	rsync -arv ./output/raw/src/generic/*.hs ./raw/src/generic/

build-torch-core:
	cd vendor && ./build-torch-core.sh

codegen-generic: build
	stack exec codegen-generic

codegen-concrete: build
	stack exec codegen-concrete

codegen-managed: build
	stack exec codegen-managed

codegen: codegen-managed codegen-concrete codegen-generic

.PHONY: clean build refresh codegen init

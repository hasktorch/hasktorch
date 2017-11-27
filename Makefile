init:
	git submodule update --init --recursive
	( cd vendor; ./build-aten.sh )
	( cd vendor; ./build-aten-spec.sh )
	( cd vendor; ./build-error-handler.sh )
	stack build
	stack test hasktorch-tests

clean:
	stack clean

build: clean
	stack build

refresh:
	rsync -arv ./output/raw/src/*.hs ./raw/src/
	rsync -arv ./output/raw/src/generic/*.hs ./raw/src/generic/

build-aten:
	cd vendor && ./build-aten.sh

build-spec:
	cd vendor && ./build-aten-spec.sh

codegen-generic: build
	stack exec codegen-generic

codegen-concrete: build
	stack exec codegen-concrete

codegen-managed: build
	stack exec codegen-managed

codegen: codegen-managed codegen-concrete codegen-generic

.PHONY: clean build refresh codegen init

UNAME:=$(shell uname)
PWD:=$(shell pwd)

# intero and stack ghci are unable to use DYLD_LIBRARY_PATH on OSX
# See https://ghc.haskell.org/trac/ghc/ticket/11617
init:
	git submodule update --init --recursive
	( cd vendor; ./build-aten.sh )
	( cd vendor; ./build-aten-spec.sh )
	( cd vendor; ./build-error-handler.sh )
# ifeq ($(UNAME),Darwin)
	ln -sf $(PWD)/vendor/build/libATen.dylib /usr/local/lib/libATen.dylib
	ln -sf $(PWD)/vendor/build/libEHX.dylib /usr/local/lib/libEHX.dylib
	@echo "\nCreated shared library symlinks for OSX:\n"
	@ls -l /usr/local/lib/libATen.dylib /usr/local/lib/libEHX.dylib
	@echo
# endif
#	stack build

clean:
	stack clean

purge: # clean
	rm -rf vendor
	git checkout -- vendor

build:
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

dev:
	sos -p '(raw-.*|core|examples)/[^.].*' -c "rm -rf dist-newstyle && cabal new-build all"

.PHONY: clean build refresh codegen init dev

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
	sudo ln -sf $(PWD)/vendor/build/libATen.dylib /usr/local/lib/libATen.dylib
	sudo ln -sf $(PWD)/vendor/build/libEHX.dylib /usr/local/lib/libEHX.dylib
	@echo "\nCreated shared library symlinks for OSX:\n"
	@sudo ls -l /usr/local/lib/libATen.dylib /usr/local/lib/libEHX.dylib
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
	rsync -arv ./output/raw/th/src/*.hs ./raw/src/
	rsync -arv ./output/raw/th/src/generic/*.hs ./raw/src/generic/

build-aten:
	cd vendor && ./build-aten.sh

build-spec:
	cd vendor && ./build-aten-spec.sh

codegen-generic: build
	cabal new-run hasktorch-codegen:ht-codegen -- --type generic --lib TH --verbose

codegen-concrete: build
	cabal new-run hasktorch-codegen:ht-codegen -- --type concrete --lib TH --verbose

codegen: codegen-concrete codegen-generic

dev:
	sos -p '(raw|core|examples)/[^.].*' -c "cabal new-build all"

dev-nuke:
	sos -p '(raw|core|examples)/[^.].*' -c "rm -rf dist-newstyle && cabal new-build all"

.PHONY: clean build refresh codegen init dev

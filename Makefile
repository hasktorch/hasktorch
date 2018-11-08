UNAME:=$(shell uname)
PWD:=$(shell pwd)
GHC_VERSION:=$(shell ghc --version | cut -d' ' -f 8)

clean:
	rm -rf dist{,-newbuild}

init:
	(cd ffi/deps && ./build-aten.sh)
	$(info GHC version detected ${GHC_VERSION})
ifeq ($(GHC_VERSION),8.4.2)
	ln -fs cabal/project.freeze-8.4.2 cabal.project.freeze
else ifeq ($(GHC_VERSION),8.4.3)
	ln -fs cabal/project.freeze-8.4.3 cabal.project.freeze
else ifeq ($(GHC_VERSION),8.4.4)
	ln -fs cabal/project.freeze-8.4.4 cabal.project.freeze
else ifeq ($(GHC_VERSION),8.6.1)
	ln -fs cabal/project.freeze-8.6.1 cabal.project.freeze
else
	$(error GHC version must be 8.4.2, 8.4.3, 8.4.4, or 8.6.1)
endif
	$(info defaulting to CPU configuration)
	./make_cabal_local.sh
	cabal new-update

purge:
	rm -rf dist-newstyle
	git submodule update --init --recursive

build:
	cabal new-build all

codegen-th:
	for l in TH THNN; do \
	  for t in generic concrete; do \
	    cabal new-run hasktorch-codegen:ht-codegen -- --type $$t --lib $$l --verbose; \
	  done; \
	done

codegen-thc:
	for l in THC THCUNN; do for t in generic concrete; do cabal new-run hasktorch-codegen:ht-codegen -- --type $$t --lib $$l --verbose; done; done

codegen: codegen-th codegen-thc

dev:
	sos -e 'dist' -p '.*hsig$$' -p '.*hs$$' -p '.*cabal$$' -p 'cabal.project$$' -c 'cabal new-build all'

run-examples:
	for ex in ad \
	  bayesian-regression \
	  download-mnist \
	  ff-typed \
	  ff-untyped \
	  gradient-descent \
	  multivariate-normal \
	  static-tensor-usage; do \
	    echo "running $$ex" && \
	    sleep 1 && \
	    cabal new-run hasktorch-examples:$$ex && \
	    sleep 1 ; \
	done
	echo "finished running examples"

dabal-all:
	for lib in \
	  "ffi/codegen/hasktorch-codegen"        \
	  "ffi/ffi/tests/hasktorch-ffi-tests"    \
	  "ffi/ffi/th/hasktorch-ffi-th"          \
	  "ffi/ffi/thc/hasktorch-ffi-thc"        \
	  "ffi/types/th/hasktorch-types-th"      \
	  "ffi/types/thc/hasktorch-types-thc"    \
	  "signatures/hasktorch-signatures"      \
	  "signatures/partial/hasktorch-signatures-partial" \
	  "signatures/support/hasktorch-signatures-support" \
	  "signatures/types/hasktorch-signatures-types"     \
	  "indef/hasktorch-indef"                           \
	  "zoo/hasktorch-zoo"; do                           \
	  $(MAKE) dabal DABAL=$${lib} & > /dev/null ; \
	done

# "examples/hasktorch-examples" # <<< this is still unstable

dabal: dabal-tmp dabal-tmp-switch

dabal-tmp:
	dhall-to-cabal $(DABAL).dhall > $(DABAL)-tmp.cabal

dabal-tmp-switch:
	mv $(DABAL)-tmp.cabal $(DABAL).cabal

test-signatures:
	for pkg in \
	  floating-th \
	  floating-thc \
	  signed-th \
	  signed-thc \
	  unsigned-thc \
	  unsigned-th; do \
	    cabal new-build hasktorch-signatures:isdefinite-$${pkg} > /dev/null ; \
	done

.PHONY: clean build refresh codegen init dev run-examples

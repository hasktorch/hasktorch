UNAME:=$(shell uname)
PWD:=$(shell pwd)

clean:
	rm -rf dist{,-newbuild}

purge:
	rm -rf vendor
	git checkout -- vendor
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

# lasso is broken
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

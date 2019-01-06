PWD := $(shell pwd)

ifeq ($(shell uname),Darwin)
LIBSUFFIX=dylib
else
LIBSUFFIX=so
endif

libtorch-test: libtorch_lib/libtorch.$(LIBSUFFIX)
	mkdir -p build
	cd build ; echo "pwd is $(PWD)"
	cd build ; cmake .. -DCMAKE_PREFIX_PATH=$(PWD)/../deps/libtorch/share/cmake/Torch/
	cd build ; make
	. ./set-dyld-path.sh ; cd build ; ./libtorch-test

libtorch_lib/libtorch.$(LIBSUFFIX):
	cd ../deps/ ; ./get-deps.sh

mkl_lib/libmklml.$(LIBSUFFIX):
	cd ../deps/ ; ./get-deps.sh

clean:
	rm -rf build
	cd ../deps/ ; ./clean-deps.sh

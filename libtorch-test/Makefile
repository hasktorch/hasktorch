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
	export LD_LIBRARY_PATH=$(PWD)/deps/mklml_mac_2019.0.1.20181227/lib:$(PWD)/deps/libtorch/lib
	. ./set-dyld-path.sh ; cd build ; ./libtorch-test

libtorch_lib/libtorch.$(LIBSUFFIX):
	cd ../deps/ ; ./get-deps.sh

mkl_lib/libmklml.$(LIBSUFFIX):
	cd ../deps/ ; ./get-deps.sh

clean:
	rm -rf build
	cd ../deps/ ; ./clean-deps.sh

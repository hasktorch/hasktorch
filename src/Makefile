PWD := $(shell pwd)

cpptest:
	mkdir -p build
	cd build ; echo "pwd is $(PWD)"
	cd build ; cmake .. -DCMAKE_PREFIX_PATH=$(PWD)/../deps/libtorch/share/cmake/Torch/
	cd build ; make
	export LD_LIBRARY_PATH=$(PWD)/deps/mklml_mac_2019.0.1.20181227/lib:$(PWD)/deps/libtorch/lib
	source ./set-dyld-path.sh ; cd build ; ./libtorch-test

clean:
	rm -rf build

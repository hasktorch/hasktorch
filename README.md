# ffi-experimental

[![master status]
(https://circleci.com/gh/hasktorch/ffi-experimental/tree/master.svg)](https://circleci.com/gh/hasktorch/ffi-experimental/tree/master)

[![dev status]
(https://circleci.com/gh/hasktorch/ffi-experimental/tree/dev.svg)](https://circleci.com/gh/hasktorch/ffi-experimental/tree/dev)

(master branch | dev branch)

Experimental next-gen code generation for aten bindings in preparation for 0.0.2 which targets the 1.0 aten backend.

Ideas being explored:

- Use yaml specs (which seemed to have been cleaned up since PT ~ 0.4) instead of header parsing.
- Try inline-cpp functionality to bind the C++ API instead of the C API. Benchmark potential template haskell overhead vs. other approaches.
- Get a vertical slice working for a small number of functions.
- Scale up.

## yaml -> binding codegen (WIP)

To run:

```
stack build ffi-experimental
stack exec ffi-experimental
```

To get CLI options:

```
stack exec ffi-experimental -- --help
```

Contributions/PRs are welcome.

## libtorch dependency retrieval and testing

`libtorch-test/` and `deps/` have scripts that retrieve libtorch and mkl-dnn library dependencies and builds/tests them using a [minimal example](https://pytorch.org/cppdocs/installing.html).

To build:

```
cd libtorch-test
make
```

The makefile pulls in libtorch + mkl-dnn (not included in libtorch) prebuilt shared library files.

If the dependencies are retrieved and built successfully, you should see at the end of the build process the expected output of a random 2x2 matrix (exact values may differ):

```
...
[ 50%] Building CXX object CMakeFiles/libtorch-test.dir/cpptest.cpp.o
[100%] Linking CXX executable libtorch-test
[100%] Built target libtorch-test
export LD_LIBRARY_PATH=/Users/huanga13/projects-personal/ffi-experimental/libtorch-test/deps/mklml_mac_2019.0.1.20181227/lib:/Users/huanga13/projects-personal/ffi-experimental/libtorch-test/deps/libtorch/lib
source ./set-dyld-path.sh ; cd build ; ./libtorch-test
 0.5790  0.5507  0.6433
 0.9908  0.6380  0.3997
[ Variable[CPUFloatType]{2,3} ]
```
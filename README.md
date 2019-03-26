# Experimental Libtorch 1.0 FFI

Experimental work on next-gen ffi bindings into the c++ libtorch library in preparation for 0.0.2 which targets the 1.0 backend.

General approach:

- Use generated `Declarations.yaml` spec instead of header parsing for code generation.
- Try inline-cpp functionality to bind the C++ API instead of the C API. Benchmark potential template haskell overhead vs. code generating C wrappers for C++ functions.

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

## yaml -> binding codegen (WIP)

To run:

```
stack build codegen
stack exec codegen-exe
```

To get CLI options:

```
stack exec codegen-exe -- --help
```

## ffi testing (WIP)

To run (currently works in the repl):

```
stack ghci --ghc-options='-fobject-code'  ffi
...
Prelude Main> main
Hello torch!
 0.2401  0.0901  0.9807
 0.9168  0.3757  0.4029
[ Variable[CPUFloatType]{2,3} ]
 1  1
 1  1
[ Variable[CPUFloatType]{2,2} ]
 0.1232  1.5721
 0.5392  0.2395
[ Variable[CPUFloatType]{2,2} ]
 1.1232  2.5721
 1.5392  1.2395
[ Variable[CPUFloatType]{2,2} ]
Prelude Main>
```

## Contributions

Contributions/PRs are welcome. 

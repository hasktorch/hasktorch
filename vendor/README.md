# vendor

Dependencies to external repositories go in this directory.

Currently pytorch is included as a submodule as the most recent changes to
`aten` and the `TH` core set of functions are maintained in the pytorch repo.

*`build-aten.sh`* is a script that builds the aten library within the pytorch
repo, including the shared `TH` library functions.

The shared library that is generated (libATen.dylib on OSX and libATen.so on
linux) is loaded and required at runtime by the hasktorch library.

*`build-aten-spec.sh`* preprocesses PyTorch's `*.cwrap` files to be yaml-compliant
formats to be use by `ProcessSpec.hs`. Use of `cwrap` files for code generation
is experimental currently so these steps are not required to be able to build or
use the library itself.

# OSX C++ Compiler Versions

Default clang versions on OSX are older and do not support modern C++
(C++11/14/17) language extensions, leading to compilation failures. This is why
the `build-aten.sh` script asks for gcc7 to be installed with homebrew.

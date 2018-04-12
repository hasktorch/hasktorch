# vendor

Dependencies to external repositories go in this directory.

Currently our fork of ATen is included as a submodule tied to the latest commit
which includes the `expand*` broadcasting functions. ATen is a mirror of the
`aten` folder in pytorch and is maintained by the pytorch community.

*`build-aten.sh`* is a script that builds the aten library within the pytorch
repo, including the shared `TH` library functions.

The shared library that is generated (libATen.dylib on OSX and libATen.so/.a on
linux) is loaded and required at runtime by the hasktorch library.

*`build-aten-spec.sh.bk`* is a backup script which preprocesses PyTorch's `*.cwrap`
files to be yaml-compliant formats to be use by `ProcessSpec.hs` in codegen. Use of
`cwrap` files for code generation was experimental and is not in active development.
Currently, all of this code is broken and it looks like a dead end, but aten support
may be rekindled if they expose a C-API.

# OSX C++ Compiler Versions

Default clang versions on OSX are older and do not support modern C++
(C++11/14/17) language extensions, leading to compilation failures. This is why
the `build-aten.sh` script asks for gcc7 to be installed with homebrew.

# hasktorch

[![Build Status](https://circleci.com/gh/hasktorch/hasktorch/tree/master.svg?style=shield&circle-token=9455d7cc953a58204f4d8dd683e9fa03fd5b2744)](https://circleci.com/gh/hasktorch/hasktorch/tree/master)

Hasktorch is a library for tensors and neural networks in Haskell. It is an independent open source community project which leverages the core C libraries shared by [Torch](http://torch.ch/) and [PyTorch](http://pytorch.org/). This library leverages cabal new-build and [backpack][backpack].

[backpack]: https://github.com/ezyang/ghc-proposals/blob/backpack/proposals/0000-backpack.rst

**Note that this project is in early development and should only be used by contributing developers. Expect substantial changes to the library API as it evolves. Contributions and PRs are welcome (see details below).**

## Project Organization

In order of high-to-low level. The reverse order can also be seen in the `cabal.project` file.

| Directory                   | Description |
| --------------------------- | ----------- |
| [`examples/`][examples]     | Examples of basic usage and experimental prototypes from a simple `hasktorch-core` dependency |
| [`core/`][core]             | Reexports of all typeclasses (see `classes/`) and instances (see `indef/`) |
| [`dimensions/`][dimensions] | Reexports and helpers of the `dimensions` library |
| [`classes/`][classes]       | Typeclasses and helpers which consist of a user-friendly Haskell API |
| [`indef/`][indef]           | Orphan instances of the above typeclasses for the relevant backpack signatures |
| [`signatures/`][signatures] | Backpack signatures which line up with the generated C-FFI |
| [`types/`][types]           | Memory-managed tensors and core data types that satisfy global and type-specific backpack types |
| [`raw/`][raw]               | Comprehensive raw bindings to C operations (TH, THNN, THC, THCUNN) |
| [`codegen/`][codegen]       | Code generation to produce low-level raw Haskell bindings |
| [`output/`][output]         | Staging directory for `codegen/` output, contents should not be under source control |
| [`vendor/`][vendor]         | 3rd party dependencies as git submodules (links to ATen and, possibly, other libraries) |

## Build Instructions 

Currently hasktorch only supports OSX and Linux builds because these are what the development team works on -- if you would like to add *BSD or Windows support, please let us know!

Building Hasktorch requires retrieving and building a fork of pytorch's [ATen library][aten] dependency. Currently (04/12/2018), we fork ATen to reintroduce some C-native broadcasting functionality which was moved to ATen's C++ core. This step has been condensed in our Makefile as `make init`.

[aten]:https://github.com/zdevito/ATen/

Following this, you will need cabal-install > 2.0 for `new-build` and backpack support. If you would like to `new-run` the examples on the command line (instead of dropping into a `new-repl`) you will need to build off of the cabal's head ([here's the current commit][cabal-latest]). If this seems intimidating, wait for [backpack support in stack][stack-backpack].

[cabal-latest]:https://github.com/haskell/cabal/tree/30d0c10349b6cc69adebfe06c230c784574ebf7a
[stack-backpack]:https://github.com/commercialhaskell/stack/issues/2540

Ensure that libATen.so is on your library path. This can be done by sourcing the `setenv` file or configuring cabal from `~/.cabal/config`. Now you can build hasktorch:

```
cabal new-build all
cabal new-run static-tensor-usage
```

For more development tips, see [DEVELOPERS.md][developers] and review the [`vendor/`][vendor] readme for details on external dependencies.

## Getting started

For an example of basic end-user API usage, see the [statically typed
tensor usage][static-tensor-usage] example and the example of [simple gradient descent][gradient-descent].

[static-tensor-usage]: https://github.com/hasktorch/hasktorch/blob/master/examples/static-tensor-usage/Main.hs
[gradient-descent]:https://github.com/hasktorch/hasktorch/blob/master/examples/gradient-descent/GradientDescent.hs

For details on implementation and usage of raw C bindings and the core library,
refer to their respective README documentation in [`raw/`][raw] and
[`core/`][core] package directories. Additional examples can be found in
[`examples/`][examples] as well as the test modules.

## Contributing

We welcome new contributors. For a rough list of outstanding items on deck
(there are many places to contribute), refer to:

https://github.com/hasktorch/hasktorch/projects/1

Contact Austin Huang or Sam Stites for access to the [hasktorch slack channel][slack]. You can find our contact information by digging through cabal files or you can ping us on gitter in the [DataHaskell Lobby][gitter-dh].

[slack]:https://hasktorch.slack.com 
[gitter-dh]:https://gitter.im/dataHaskell/Lobby

## Thanks

Thanks to all hasktorch developers who have contributed to this community effort so far. This project is also indebted to prior work on typed functional
programming for deep learning by
Justin Le [![][gh-icon]][mstkg-gh][![][blog-icon]][mstkg],
Edward Yang [![][gh-icon]][ezyang-gh][![][blog-icon]][ezyang],
Huw Campbell [![][gh-icon]][huw-gh][![][blog-icon]][huw],
Kaixi Ruan [![][gh-icon]][krpopo-gh],
and Khanh Nguyen [![][gh-icon]][khanhptnk-gh],
as well as to the Torch and PyTorch dev teams.

[gh-icon]:https://png.icons8.com/small/1x/github.png
[blog-icon]:https://png.icons8.com/small/1x/blog.png
[mstkg]:https://blog.jle.im/
[mstkg-gh]:https://github.com/mstksg
[huw-gh]:https://github.com/HuwCampbell
[huw]:http://www.huwcampbell.com/
[krpopo-gh]:https://github.com/krpopo
[khanhptnk-gh]:https://github.com/khanhptnk
[ezyang-gh]:https://github.com/ezyang/
[ezyang]:http://ezyang.com/

<!-- project directory links -->
[developers]: ./DEVELOPERS.md
[makefile]: ./Makefile
[codegen]: ./codegen/
[types]: ./types/
[signatures]: ./signatures/
[core]: ./core/
[examples]: ./examples/
[output]: ./output/
[raw]: ./raw/
[vendor]: ./vendor/
[classes]: ./classes/
[dimensions]: ./dimensions/
[indef]: ./indef/

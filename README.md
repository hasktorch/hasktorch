# hasktorch

[![Build Status](https://circleci.com/gh/austinvhuang/hasktorch/tree/master.svg?style=shield&circle-token=:9455d7cc953a58204f4d8dd683e9fa03fd5b2744)](https://circleci.com/gh/austinvhuang/hasktorch/tree/master)

A Haskell library for tensors and neural networks. Hasktorch uses the low-level
[TorcH](https://github.com/pytorch/pytorch/tree/master/torch/lib/TH) C
tensor/math library shared by [Torch](http://torch.ch/) and
[PyTorch](http://pytorch.org/). This project is an independent
(unsponsored/unaffiliated) open source effort.

**NOTE: hasktorch is in early development and should only be used by
contributing developers at the current time.**

## Project Structure

| Directory | Description |
| --------- | ----------- |
| [`codegen/`][codegen] | Parsers that parse .h files in the C TH library and generate low-level raw Haskell bindings.
| [`core/`][core] | Memory-managed tensors and core data types that wrap raw C bindings to TH.
| [`nn/`][nn] | Neural network library (not much here atm)
| [`interface/`][interface] | Shared types
| [`output/`][output] | Staging directory for `codegen/` output, contents should not be under source control.
| [`raw/`][raw] | Comprehensive raw bindings to several hundred C TorcH (TH) operations, including separate bindings for all tensor types.
| [`tests/`][tests] | Continuous integration tests
| [`tools/`][tools] | Misc tools
| [`vendor/`][vendor] | 3rd party dependencies as git submodules (links to TH C and other libraries)

## Build Instructions

Currently building hasktorch is only supported on OSX and linux. To start,
retrieve git submodules (includes TorcH library) with:

```
git submodule update --init --recursive
```

A recent version of the gcc C compiler is used to build the TorcH C library. If
gcc is already installed, you should be able to run this shell script from
within the `vendor/` directory:

```
cd vendor; ./build-torch-core.sh
```

On OSX the above script looks for the gcc-6 binary which needs to be installed
using [homebrew](https://brew.sh/). On linux, this simply uses gcc. 

If successful, this builds the TH shared library and places it in the
`vendor/build/` directory. Then build the project using stack:

```
stack build
```

If everything built, you should be able to run tests successfully:

```
stack test hasktorch-tests
```

## Code Generation

The [`raw/`][raw] modules are generated using the scripts in
[`codegen/`][codegen]. Since the outputs are already part of the repo, you
should not need to run [`codegen/`][codegen] programs to use hasktorch.

However, if you are contributing to hasktorch itself, you may want to
modify/re-generate the code generation processes. Currently there are three main
operations:

- `stack exec codegen-generic` - Builds generic modules (one version per tensor type).
- `stack exec codegen-concrete` - Builds non-generic modules.
- `stack exec codegen-managed` - Build memory-managed interfaces to TH (not working yet).

All of these programs write `.hs` files into the [`output/`][output] directory
as a staging area (rather than overwriting [`raw/`][raw] directly).

For details on the TH library's pseudo-templating preprocessor mechanism for
underlying the generic modules, see [Adam Paszke's
writeup](https://apaszke.github.io/torch-internals.html).

## Contributing

Contributions are welcome. For a list of things that need to get done, see:

https://github.com/austinvhuang/hasktorch/projects/1

## References

### Torch Internals

- [Torch Internals Writeup by Adam Paszke](https://apaszke.github.io/torch-internals.html) 
- [A Tour of PyTorch Internals (Part I)](http://pytorch.org/2017/05/11/Internals.html)
- [Haskell/Torch binding using backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).

###  Background on Dependent types in Haskell and NN Implementations

- [Practical dependent types in Haskell](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html)
- [Monday Morning Haskell: Deep Learning and Deep Types: Tensor Flow and Dependent Types](https://mmhaskell.com/blog/2017/9/11/deep-learning-and-deep-types-tensor-flow-and-dependent-types)
- [Basic Type Level Programming in Haskell](http://www.parsonsmatt.org/2017/04/26/basic_type_level_programming_in_haskell.html)
- [static dimension checking](http://dis.um.es/~alberto/hmatrix/static.html)
- [defunctionalization](https://typesandkinds.wordpress.com/2013/04/01/defunctionalization-for-the-win/)
- [An introduction to DataKinds and GHC.TypeLits](http://ponies.io/posts/2014-07-30-typelits.html)
- [Applying Type-Level and Generic Programming in Haskell](https://www.cs.ox.ac.uk/projects/utgp/school/andres.pdf)



### Automatic Differentiation

- [Automatic Propagation of Uncertainty with AD](https://blog.jle.im/entry/automatic-propagation-of-uncertainty-with-ad.html)
- [Automatic Differentiation is Trivial in Haskell](http://www.danielbrice.net/blog/2015-12-01/])


<!-- project directory links -->

[codegen]: ./codegen/
[core]: ./core/
[examples]: ./examples/
[interface]: ./interface/
[nn]: ./nn/
[output]: ./output/
[raw]: ./raw/
[tests]: ./tests/
[tools]: ./tools/
[vendor]: ./vendor/



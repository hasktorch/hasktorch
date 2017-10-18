# hasktorch

A Haskell library for tensors and neural networks using
[Torch](https://github.com/torch/torch7).

*This library is in early development and is not recommended for use except by
contributors*

## Project Structure

| Directory | Description |
| --------- | ----------- |
| [`codegen/`][codegen] | Parsers that wrap TH C files and generate raw Haskell bindings.
| [`core/`][core] | Memory-managed low-level operations that wrap raw bindings.
| [`examples/`][examples] | Usage examples
| [`interface/`][interface] | Shared types
| [`output/`][output] | staging directory for `codegen/` output, contents should not be under source control.
| [`raw/`][raw] | Comprehensive raw bindings to C TorcH (TH) functions.
| [`tests/`][tests] | Continuous integration tests
| [`tools/`][tools] | Misc tools
| [`vendor/`][vendor] | 3rd party dependencies as git submodules (links to TH C library)

[codegen]: ./codegen/
[core]: ./core/
[examples]: ./examples/
[interface]: ./interface/
[output]: ./output/
[raw]: ./raw/
[tests]: ./tests/
[tools]: ./tools/
[vendor]: ./vendor/

## Build Instructions

Currently building hasktorch is only supported on OSX and linux. To start,
retrieve git submodules (includes TorcH library) with:

```
git submodule update --init --recursive`
```

A recent version of the gcc C compiler is used to build the TorcH C library. If
gcc is already installed, you should be able to run this shell script from
within the `vendor/`, directory:

```
build-torch-core.sh
```

If successful, this builds the TH shared library and places it in the
`vendor/build/` directory.

Then build the project using stack:

```
stack build
```

If everything built, you should be able to run tests successfully:

```
stack test torch-tests:test-torch
stack test torch-tests:memory-test
stack test torch-tests:tensor-tests
```

## Code Generation

The `raw/` modules are generated using the scripts in `codegen`. Since the
outputs are already included in the repo, you should not need to run `codegen`
programs to use hasktorch.

However, if you are contributing to hasktorch itself, you may want to
modify/re-generate the code generation processes. Currently there are three main
operations:

- `stack exec codegen-generic` - Builds generic modules (one version per tensor type).
- `stack exec codegen-concrete` - Builds non-generic modules.
- `stack exec codegen-managed` - Build memory-managed interfaces to TH (not working yet).

## References

### Torch Internals

- [Torch Internals Writeup by Adam Paszke](https://apaszke.github.io/torch-internals.html) 
- [A Tour of PyTorch Internals (Part I)](http://pytorch.org/2017/05/11/Internals.html)
- [Haskell/Torch binding using backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).

###  Dependent type NN Implementations

- [Practical dependent types in Haskell](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html)
- [Monday Morning Haskell: Deep Learning and Deep Types: Tensor Flow and Dependent Types](https://mmhaskell.com/blog/2017/9/11/deep-learning-and-deep-types-tensor-flow-and-dependent-types)
- ["Dependent Types in Haskell" by Stephanie Weirich](https://www.youtube.com/watch?v=wNa3MMbhwS4)

### Automatic Differentiation

- [Automatic Propagation of Uncertainty with AD](https://blog.jle.im/entry/automatic-propagation-of-uncertainty-with-ad.html)
- [Automatic Differentiation is Trivial in Haskell](http://www.danielbrice.net/blog/2015-12-01/])

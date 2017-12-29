# hasktorch

[![Build Status](https://circleci.com/gh/austinvhuang/hasktorch/tree/master.svg?style=shield&circle-token=9455d7cc953a58204f4d8dd683e9fa03fd5b2744)](https://circleci.com/gh/austinvhuang/hasktorch/tree/master)

Hasktorch is a library-in-development for tensors and neural networks in
Haskell. It is an independent open source community project which leverages the
core C libraries shared by [Torch](http://torch.ch/) and
[PyTorch](http://pytorch.org/).

**This project is in early development and should only be used by contributing
developers. Expect substantial changes to the library API as it evolves.
Contributions and PRs are welcome (see details below).**

## Project Organization

| Directory | Description |
| --------- | ----------- |
| [`codegen/`][codegen] | Code generation to produce low-level raw Haskell bindings. Also includes experimental aten cwrap file parsing.
| [`core/`][core] | Memory-managed tensors and core data types that wrap raw C bindings to TH.
| [`examples/`][examples] | Examples of basic usage and experimental prototypes |
| [`output/`][output] | Staging directory for `codegen/` output, contents should not be under source control.
| [`raw/`][raw] | Comprehensive raw bindings to C TorcH (TH) operations.
| [`vendor/`][vendor] | 3rd party dependencies as git submodules (links to TH C and other libraries)

## Build Instructions 

Currently hasktorch only supports OSX and Linux builds. Building Hasktorch
requires retrieving and building C++ torch dependencies as submodules followed
by building with the [Stack](https://docs.haskellstack.org/en/stable/README/)
tool. These steps are done automatically using the [Makefile][makefile]:

```
make init
```

For manual steps to build the project, see the [developer guide][developers] for
instructions and the [`vendor/`][vendor] README for details on external
dependencies.

## Getting started

For an example of basic end-user API usage, see the example on [statically typed
tensor usage
](https://github.com/austinvhuang/hasktorch/blob/master/examples/static-tensor-usage/StaticTensorUsage.hs)
and the [simple gradient descent
example](https://github.com/austinvhuang/hasktorch/blob/master/examples/gradient-descent/GradientDescent.hs).

For details on implementation and usage of raw C bindings and the core library,
refer to their respective README documentation in [`raw/`][raw] and
[`core/`][core] package directories. Additional examples can be found in
[`examples/`][examples] as well as the test modules.

## Contributing

We welcome new contributors. For a rough list of outstanding items on deck
(there are many places to contribute), refer to:

https://github.com/austinvhuang/hasktorch/projects/1

Contact Austin Huang for access to the private hasktorch slack channel at:

https://hasktorch.slack.com 

Thanks to all hasktorch developers who have contributed to this community effort
so far. This project is also indebted to prior work on typed functional
programming for deep learning by Justin Le, Huw Campbell, and Kaixi Ruan, as
well as to the Torch and PyTorch frameworks.

<!-- project directory links -->
[developers]: ./DEVELOPERS.md
[makefile]: ./Makefile
[codegen]: ./codegen/
[core]: ./core/
[examples]: ./examples/
[output]: ./output/
[raw]: ./raw/
[vendor]: ./vendor/

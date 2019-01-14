# Information for Developers

## Learning Materials

### Introductory Materials

- Presentation video from ICFP 2018 https://www.youtube.com/watch?v=SjxP1NpoP2c
- Hasktorch conference poster fromm PyTorch Developers 2018 https://docs.google.com/presentation/d/12jxRCzWf7nYpNYyOGnPMHoz2v2O6M9I-JOKeS9ybao8/edit?usp=sharing
- Presentation slides from ICFP 2018 https://docs.google.com/presentation/d/1EuSx4RupbfG8NEOEYkfICidiVhXDGwAm5DL1PlezWkg/edit?usp=sharing
- Sam’s medium post w/ a simple xor example https://medium.com/@stites/hasktorch-v0-0-1-28d9ab270f3f

### Getting comfortable with type-level representations

- Matt Parson’s blog post Basic Type Level Programming https://www.parsonsmatt.org/2017/04/26/basic_type_level_programming_in_haskell.html - there aren’t many materials out there on type level stuff and most of them are rather advanced. I think this is the most accessible starting point.
- @Justin Le’s blog https://blog.jle.im/entries.html all the articles are great but particularly relevant are “Practical Dependent Types in Haskell” (w/ neural networks as an application, “Introducing the Backprop Library”, and “A Purely Functional Typed Approach to Trainable Models”

### Backpack

- [Backpack to Work: Towards Backpack in Practice](https://www.youtube.com/watch?v=A3ehG4GQpxU)

### Applications and usage of the high level API

- static-tensor-usage and backprop-regression under https://github.com/hasktorch/hasktorch/tree/master/examples
- For background on the backprop abstraction, see both Justin’s blog post on backprop and the documentation https://backprop.jle.im/

## Building Hasktorch Manually

To start, retrieve git submodules (includes TorcH library) with:

```
git submodule update --init --recursive
```

A recent version of the gcc C compiler is used to build the TorcH C library. If
gcc is already installed, you should be able to run these scripts from
within the `vendor/` directory:

```
cd vendor
./build-aten.sh
./build-error-handler.sh
```

the `build-aten.sh` script builds `libATen.so`, while the
`build-error-handler.sh` script builds libEHX.so (or .dylib on OSX). Currently
the error handler isn't fully implemented, but developers can refer to
[Exceptions.hs](https://github.com/austinvhuang/hasktorch/blob/master/core/src/Torch/Core/Exceptions.hs)
for an example of how the custom exception handling is integrated with the TH
API.

On OSX the above script looks for the gcc-6 binary which needs to be installed
using [homebrew](https://brew.sh/). On linux, this simply uses gcc. 

If successful, this builds the TH shared library and places it in the
`vendor/build/` directory. Then build the project using stack:

```
cabal new-build all
```

If everything built, you should be able to run tests successfully:

```
cabal new-test all
```

# Common Issues

## Building Hasktorch

### libnuma-dev install

### cabal install issues

Cabal installs in odd locations & it's quite some effort to get cabal 2.2.4 installed

### python dependencies

pyyaml
typing

### `cabal: Could not resolve dependencies:...`

usually resolved by having compatable GHC and cabal-install versions.

For our own builds, we currently use ghc 8.4.3 and cabal-install 2.2.0.0 or cabal-install 2.3.0.0.

### CPU-based Compute, Missing CUDA Dependencies

If you're running on a machine without a GPU, the cuda flag should be turned off. 

This can be done on the command line, for example:

```
cabal new-build all --flags=-cuda
```

Or for running a specific example:

```
cabal new-run lenet-cifar10 --flags=-cuda
```

Alternatively, cabal settings can be set using [cabal.project.local](https://www.haskell.org/cabal/users-guide/nix-local-build.html#configuring-builds-with-cabal-project).

## Other References

### Torch Internals

- [Torch Internals Writeup by Adam Paszke](https://apaszke.github.io/torch-internals.html) 
- [A Tour of PyTorch Internals (Part I)](http://pytorch.org/2017/05/11/Internals.html)
- [Haskell/Torch binding using backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).

###  Background on Dependent types in Haskell and NN Implementations

- [Practical dependent types in Haskell](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html)
- [Monday Morning Haskell: Deep Learning and Deep Types: Tensor Flow and Dependent Types](https://mmhaskell.com/blog/2017/9/11/deep-learning-and-deep-types-tensor-flow-and-dependent-types)
- [static dimension checking](http://dis.um.es/~alberto/hmatrix/static.html)
- [defunctionalization](https://typesandkinds.wordpress.com/2013/04/01/defunctionalization-for-the-win/)
- [An introduction to DataKinds and GHC.TypeLits](http://ponies.io/posts/2014-07-30-typelits.html)
- [Applying Type-Level and Generic Programming in Haskell](https://www.cs.ox.ac.uk/projects/utgp/school/andres.pdf)

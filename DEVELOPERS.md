# Information for Developers

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
- [Automatic Differentiation is Trivial in Haskell](http://www.danielbrice.net/blog/2015-12-01/)

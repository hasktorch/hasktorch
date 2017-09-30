# HaskTorch

A Haskell library for tensors and dynamic neural networks using
[Torch](https://github.com/torch/torch7).

*This library is in early development and is not recommended for use except by
project contributors*

## Setup Instructions

Retrieve git submodules with

```
git submodule update --init --recursive`
```

From within the `vendor`, directory, run:

```
build-torch-core.sh
```

Then build using stack:

```
stack build
```

If everything built, you should be able to run tests successfully:

```
stack test torch-tests
```

## References

- [torch internals](https://apaszke.github.io/torch-internals.html) 
- [A Tour of PyTorch Internals (Part I)](http://pytorch.org/2017/05/11/Internals.html)
- [Haskell/Torch binding using backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).
- [Practical dependent types in Haskell](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html)

